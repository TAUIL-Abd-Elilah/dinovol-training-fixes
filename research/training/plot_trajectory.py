"""Render measured paired trajectories; no smoothing or quality inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('comparison',type=Path)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
d=json.loads(args.comparison.read_text())
s=d['steps'];x=[v['number']+1 for v in s]
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
fig.suptitle('Dinovol: 100 paired updates on changing real CT crops',fontsize=15)
a=axes[0,0]
a.plot(x,[v['loss']['loss']['native'] for v in s],label='Native',color='#193D6B')
a.plot(x,[v['loss']['loss']['candidate'] for v in s],label='FP32 padded',color='#D87319',linestyle='--')
a.set(title='Total training loss',ylabel='Loss');a.legend()
a=axes[0,1]
a.plot(x,[abs(v['loss']['loss']['absolute_delta']) for v in s],color='#863A76')
a.set(title='Absolute loss difference',ylabel='Absolute difference')
a=axes[1,0]
for key,label,color in [('gradient','Gradient','#193D6B'),('student_delta','AdamW update','#D87319')]:
    a.plot(x,[100*v['samples'][key]['relative_l2'] for v in s],label=label,color=color)
a.set(title='Sampled step differences',ylabel='Relative L2 difference (%)');a.legend()
a=axes[1,1]
for key,label,color in [('student_after','Student','#193D6B'),('teacher_after','Teacher','#D87319')]:
    a.plot(x,[100*v['samples'][key]['relative_l2'] for v in s],label=label,color=color)
a.set(title='Sampled accumulated parameter differences',ylabel='Relative L2 difference (%)');a.legend()
for a in axes.flat:
    a.set_xlabel('Completed optimizer update');a.grid(alpha=.2)
    a.axvspan(.5,5.5,color='gray',alpha=.10)
    a.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
fig.supxlabel('Same source, seed and inputs; only attention option differs. Shaded: warmup.\n'
              '58,240 sampled values per update; full gradient finiteness checked. No convergence or ink-accuracy claim.',fontsize=9)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=160)
plt.close(fig)
