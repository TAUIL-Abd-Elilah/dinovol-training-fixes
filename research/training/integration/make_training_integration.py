"""Generate and application-check the opt-in official Dinovol training patch."""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PIN = 'f07d33be6a00d12ace7d6a9465efe17c78ed7b47'
PREFIX = 'dinovol/dinovol_2/model/'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise AssertionError('Expected exactly one anchor: ' + old)
    return source.replace(old, new, 1)


def sources(villa):
    subprocess.run(['git','-C',str(villa),'diff','--exit-code',PIN,'--','dinovol/dinovol_2'],
                   check=True,capture_output=True)
    paths = [PREFIX+'dinov2_eva.py', PREFIX+'model.py']
    original = {path:subprocess.check_output(['git','-C',str(villa),'show',PIN+':'+path],text=True)
                for path in paths}
    eva = original[paths[0]]
    eva = replace_once(eva,
        'from dinovol_2.model.patch_encode_decode import PatchEmbed, PatchEmbedDeeper\n',
        'from dinovol_2.model.patch_encode_decode import PatchEmbed, PatchEmbedDeeper\nfrom dinovol_2.model.training_sdpa_padding import padded_training_sdpa\n')
    eva = replace_once(eva, '            norm_layer: Optional[Callable] = None,\n    ):',
                      '            norm_layer: Optional[Callable] = None,\n            pad_sdpa_heads: bool = False,\n    ):')
    eva = replace_once(eva, '        self.fused_attn = use_fused_attn()\n',
                      '        self.fused_attn = use_fused_attn()\n        if not isinstance(pad_sdpa_heads, bool):\n            raise TypeError("pad_sdpa_heads must be a boolean")\n        self.pad_sdpa_heads = pad_sdpa_heads\n')
    old = '''            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )'''
    new = '''            if self.pad_sdpa_heads and not torch.jit.is_scripting():
                x = padded_training_sdpa(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                    enabled=True,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )'''
    eva = replace_once(eva,old,new)
    eva = replace_once(eva, '            ndim: Optional[int] = None,\n    ):',
                      '            ndim: Optional[int] = None,\n            pad_sdpa_heads: bool = False,\n    ):')
    eva = replace_once(eva, '            norm_layer=norm_layer if scale_attn_inner else None,\n',
                      '            norm_layer=norm_layer if scale_attn_inner else None,\n            pad_sdpa_heads=pad_sdpa_heads,\n')
    eva = replace_once(eva, '            deeper_embed_batch_chunk_size: Optional[int] = None,\n    ):',
                      '            deeper_embed_batch_chunk_size: Optional[int] = None,\n            pad_sdpa_heads: bool = False,\n    ):')
    eva = replace_once(eva, '                ndim=self.ndim,\n            )\n            for i in range(depth)])',
                      '                ndim=self.ndim,\n                pad_sdpa_heads=pad_sdpa_heads,\n            )\n            for i in range(depth)])')
    model = original[paths[1]]
    for value in ('False','True'):
        model = replace_once(model, '    "qkv_fused": '+value+',\n',
                             '    "qkv_fused": '+value+',\n    "pad_sdpa_heads": False,\n')
    updated = {paths[0]:eva,paths[1]:model,
               PREFIX+'training_sdpa_padding.py':(ROOT/'training_sdpa_padding.py').read_text(encoding='utf-8')}
    for path,source in updated.items():
        ast.parse(source,filename=path)
    return original,updated


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--villa',type=Path,required=True)
    args = ap.parse_args()
    original,updated = sources(args.villa)
    pieces = []
    for path,source in updated.items():
        pieces.append('diff --git a/'+path+' b/'+path+'\n')
        if path not in original:
            pieces.append('new file mode 100644\n')
        pieces.extend(difflib.unified_diff(original.get(path,'').splitlines(keepends=True),
            source.splitlines(keepends=True),fromfile='a/'+path if path in original else '/dev/null',tofile='b/'+path))
    target = ROOT/'dinovol_training_head_padding.patch'
    target.write_text(''.join(pieces),encoding='utf-8',newline='\n')
    subprocess.run(['git','-C',str(args.villa),'apply','--check',str(target)],check=True,capture_output=True)
    receipt = {'status':'AST_AND_PATCH_CHECK_PASS','source_pin':PIN,'checkout_edited':False,
               'patch_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
               'original_normalized_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in original.items()},
               'updated_normalized_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in updated.items()},
               'helper_sha256':hashlib.sha256((ROOT/'training_sdpa_padding.py').read_bytes()).hexdigest()}
    (ROOT/'PATCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
