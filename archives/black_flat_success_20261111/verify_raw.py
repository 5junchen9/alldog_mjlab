"""Independent raw-trajectory arithmetic and resource verification, CPU only."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'logs/black_teacherfree/extra_seeds_20261111'
plan=json.loads((OUT/'plan.json').read_text())
records=[]
initial=[]
for seed in plan['protocol']['seeds']:
    for mode in plan['protocol']['modes']:
        pair=[]
        for policy in ['final','omni']:
            stem=OUT/policy/f'seed_{seed}_{mode}'
            report=json.loads(stem.with_suffix('.json').read_text())
            audit=json.loads(stem.with_suffix('.config.json').read_text())
            raw=np.load(stem.with_name(f'{stem.name}_seed{seed}_raw.npz'))
            assert report['complete'] and report['checkpoint_unchanged'] and report['actor_and_normalizer_unchanged']
            assert report['checkpoint']==plan['models'][policy]
            assert report['seed']==seed and report['mode']==mode and report['num_envs']==224
            assert len(report['rows'])==28
            nfall=0;previous_valid=np.ones(224,bool);previous_cmd=np.zeros((224,3),np.float32)
            raw_falls=[]
            for phase,length,sign in [('settle',100,0),('command',300,1),('reverse',300,-1),('stop',250,0)]:
                v=raw[f'{phase}_velocity'];c=raw[f'{phase}_command'];valid=raw[f'{phase}_valid']
                goal=np.repeat(np.array([s['command'] for s in plan['protocol']['scenarios']],dtype=np.float32),8,axis=0)*sign
                assert v.shape==(length,224,3) and c.shape==v.shape and valid.shape==(length,224)
                assert np.isfinite(v).all() and np.isfinite(c).all()
                assert (valid[0]<=previous_valid).all() and (valid[1:]<=valid[:-1]).all()
                assert np.array_equal(raw[f'{phase}_new_falls'],previous_valid & ~valid[-1])
                assert np.allclose(raw[f'{phase}_error'],np.abs(v-goal),atol=1e-6,rtol=1e-6)
                if mode=='abrupt':
                    assert np.allclose(c,goal,atol=1e-7)
                else:
                    changes=np.diff(np.concatenate([previous_cmd[None],c]),axis=0)
                    assert np.linalg.norm(changes[:,:,:2],axis=2).max()<=.020001
                    assert np.abs(changes[:,:,2]).max()<=.010001
                    assert np.allclose(c[-1],goal,atol=1e-6)
                previous_valid=valid[-1];previous_cmd=c[-1]
                raw_falls.append(int(raw[f'{phase}_new_falls'].sum()))
            assert report['trials_with_fall']==int((~previous_valid).sum())==sum(raw_falls)
            for i,row in enumerate(report['rows']):
                sl=slice(i*8,(i+1)*8)
                assert row['replicates']==8 and row['target_command']==plan['protocol']['scenarios'][i]['command']
                assert row['trials_with_fall']==int((~previous_valid[sl]).sum())
                for phase in ['settle','command','reverse','stop']:
                    assert row['phase_first_falls'][phase]==int(raw[f'{phase}_new_falls'][sl].sum())
                for phase in ['command','reverse']:
                    mask=raw[f'{phase}_valid'][150:,sl]
                    errors=raw[f'{phase}_error'][150:,sl]
                    p=row[phase]
                    assert p['valid_samples']==int(mask.sum()) and p['total_samples']==1200
                    if mask.any():
                        assert np.allclose(p['mae_xyz'],errors[mask].mean(axis=0),atol=1e-7,rtol=1e-6)
                    else:
                        assert p['mae_xyz'] is None
            schedule_hash=hashlib.sha256(b''.join(raw[f'{p}_command'].tobytes() for p in ['settle','command','reverse','stop'])).hexdigest()
            pair.append({'policy':policy,'cfg':audit['config_sha256'],'init':audit['initialization_sha256'],'commands':schedule_hash})
            records.append({'policy':policy,'seed':seed,'mode':mode,'falls':report['trials_with_fall'],'raw_arithmetic_verified':True,'elapsed_seconds':report['finished_at']-report['started_at']})
        assert pair[0]['cfg']==pair[1]['cfg'] and pair[0]['init']==pair[1]['init'] and pair[0]['commands']==pair[1]['commands']
        initial.append({'seed':seed,'mode':mode,'matched_config':True,'matched_initial_physics_state':True,'matched_full_command_trajectory':True})
samples=[json.loads(l) for l in (OUT/'gpu_samples.jsonl').read_text().splitlines()]
max_jobs=0;temperature=[];used=[];free=[]
for s in samples:
    jobs=[line for line in s['compute'].splitlines() if line.strip() and '/usr/libexec/gnome-remote-desktop-daemon' not in line]
    max_jobs=max(max_jobs,len(jobs))
    mem,fr,temp,util=map(int,s['gpu'].split(','));used.append(mem);free.append(fr);temperature.append(temp)
assert max_jobs<=1
# Independently check the finalized summary from its per-seed inputs.
summary=json.loads((OUT/'summary.json').read_text())
for mode,groups in summary['groups'].items():
    for group,entry in groups.items():
        for name in ['models','paired_final_minus_omni']:
            stats_list=entry[name].values() if name=='models' else [entry[name]]
            for stats_group in stats_list:
                for key,stats in stats_group.items():
                    for d in stats if isinstance(stats,list) else [stats]:
                        v=np.array(d['values']);assert len(v)==10 and np.isfinite(v).all()
                        assert abs(float(v.mean())-d['mean'])<1e-12
                        assert abs(float(v.var(ddof=1))-d['sample_variance'])<1e-12
        for key in ['fall_fraction','tracking_mae','valid_sample_fraction','response_ratio_mean','reversal_success_fraction','stop_within_3s_fraction']:
            a=np.array(entry['models']['final'][key]['values']);b=np.array(entry['models']['omni'][key]['values'])
            assert np.allclose(a-b,entry['paired_final_minus_omni'][key]['values'],atol=1e-12)
result={'complete':True,'raw_runs_verified':40,'trials_verified':8960,'raw_checks':records,'paired_checks':initial,'gpu':{'max_concurrent_non_desktop_compute_jobs':max_jobs,'samples':len(samples),'peak_used_mib':max(used),'minimum_free_mib':min(free),'maximum_temperature_c':max(temperature)},'summary_arithmetic_verified':True}
(ROOT/'RECHECK_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['raw_checks','paired_checks']},indent=2))
