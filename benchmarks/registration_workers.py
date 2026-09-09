"""Bounded real registration executor benchmark, with identical FITS checks.

Run with the repository Python. No app launch or end-to-end processing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time

from astropy.io import fits
import numpy as np

from openastroflow_engine import calibration, pixel_pipeline as pipeline
from registration_lanczos_kernel import load_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=1536)
    parser.add_argument('--width', type=int, default=4096)
    args = parser.parse_args()
    original_counter = pipeline._registration_worker_count
    candidate = calibration.FitsFrame.sample_lanczos3_clamped
    baseline, baseline_source = load_baseline(args.baseline_source)
    budget = 2 * 1024**3
    report = {'shape': [args.height, args.width], 'frames': 8,
              'budgetBytes': budget, 'evidenceClass': 'isolated-synthetic-affine-warp',
              'baselineFunctionSha256': hashlib.sha256(baseline_source.encode()).hexdigest(),
              'numpyVersion': np.__version__, 'measurements': []}
    reference_hashes = None
    with tempfile.TemporaryDirectory(prefix='wbpp-registration-workers-') as work:
        root = Path(work)
        source = root / 'source.fits'
        pixels = np.random.default_rng(875).normal(0.15, 0.025, (args.height,args.width)).astype(np.float32)
        pixels[123,321] = np.nan
        fits.writeto(source,pixels,fits.Header({'IMAGETYP':'Calibrated Light','OAFNDOM':'NORMALIZED_TEST','OAFNSCL':1.0}))
        del pixels
        # This generated fixture has an explicitly known unit scale. Production
        # obtains the equivalent authority from its calibration provenance.
        info = replace(calibration.read_frame_info(source),
                       numeric_domain_authority='CONTENT_BOUND_OVERRIDE')
        matrices = []
        for index in range(8):
            angle = np.deg2rad(0.44 if index % 2 == 0 else 179.56)
            c,s = np.cos(angle),np.sin(angle)
            cx,cy = (args.width-1)/2,(args.height-1)/2
            matrices.append(pipeline.AffineTransform.from_value(((c,-s,cx-c*cx+s*cy+0.13*index),(s,c,cy-s*cx-c*cy+0.07*index),(0,0,1))))
        # Interleave repeated 4/8-worker runs to expose thermal/order noise.
        for label,workers in [('baseline',4),('candidate',4),('candidate',8),('candidate',6),('candidate',8),('candidate',4)]:
            index = len(report['measurements'])
            folder = root / str(index)
            folder.mkdir()
            jobs = [pipeline._RegistrationJob(source,folder/f'{i}.fits',matrix,info) for i,matrix in enumerate(matrices)]
            calibration.FitsFrame.sample_lanczos3_clamped = baseline if label=='baseline' else candidate
            pipeline._registration_worker_count = lambda jobs, *,max_memory_bytes,resampler,cpu_workers: min(workers,cpu_workers,len(jobs),max(1,max_memory_bytes//(args.width*192)))
            start=time.perf_counter()
            pipeline._register_frames(jobs,max_memory_bytes=budget,resampler='lanczos-3-clamped',cpu_workers=workers)
            seconds=time.perf_counter()-start
            hashes=[hashlib.sha256(job.destination.read_bytes()).hexdigest() for job in jobs]
            if reference_hashes is None: reference_hashes=hashes
            assert hashes==reference_hashes, 'Registered FITS changed with kernel/concurrency'
            item={'kernel':label,'workers':workers,'seconds':seconds,'fitsByteIdentical':True}
            report['measurements'].append(item)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(item),flush=True)
            for job in jobs: job.destination.unlink()
    pipeline._registration_worker_count=original_counter
    calibration.FitsFrame.sample_lanczos3_clamped=candidate


if __name__ == '__main__':
    main()
