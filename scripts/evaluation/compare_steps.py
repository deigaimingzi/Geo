import json,glob,os,re
D='output/v4/eval/'
rows=[('step0',json.load(open(D+'step0_init_val.json')))]
for f in sorted(glob.glob(D+'step*_val.json'),key=lambda x:int(re.findall(r'step(\d+)_',x)[0])): rows.append((os.path.basename(f)[:-9],json.load(open(f))))
if os.path.exists(D+'final_val.json'): rows.append(('final',json.load(open(D+'final_val.json'))))
ks=[('self_mae_current','self_ref_current'),('cross_mae_current','cross_ref_current'),('cross_mae_history','cross_ref_history'),('pair_dist_err','pair_dist_ref')]
print(f"{'':8s}"+''.join(f'{k[0][:19]:>21s}' for k in ks))
for n,r in rows: print(f'{n:8s}'+''.join(f'{r[a]:12.3f} /{r[b]:6.3f}' for a,b in ks))
print('(model / constant-baseline; model < baseline = learned beyond the mean)')
