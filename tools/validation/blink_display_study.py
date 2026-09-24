#!/usr/bin/env python3
"""Read-only, real-image comparison of experimental Blink display algorithms.

Consumes a content-bound existing Blink manifest; produces a create-only local
HTML gallery and share-safe metric table. No admission or science run is made.
Master calibration is deliberately required to avoid testing an optical
vignetting artifact as if it were a normalization defect. FITS native crops are
optional evidence; this study does not implement native XISF/OSC crop decoding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from lightframeqc.content_hash import file_sha256, stat_identity
from lightframeqc.readers import read_frame_preview
from ufwbpp.blink_native_crops import native_atlas, star_positions
from ufwbpp.blink_diagnostics import DISPLAY_ALGORITHM, DisplayReference, choose_display_reference, detail_transfer, diagnostic_preview, local_noise
from ufwbpp.blink_previews import _master_preview, PreviewCalibration, calibrate_linear, channel_stretch, compose_to_reference, stretch_to_8bit, warp_to_reference


def save_image(path: Path, values: np.ndarray, size: tuple[int, int] | None = None) -> None:
    image = Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))
    if size:
        image.thumbnail(size, Image.Resampling.BOX)
    image.save(path)


def matrix(frame: dict) -> np.ndarray | None:
    value = frame["transformToReference"]
    return None if value is None else np.vstack([np.asarray(value), [0, 0, 1]])



def run(args: argparse.Namespace) -> dict:
    start=time.perf_counter()
    manifest_path=args.manifest.resolve()
    manifest=json.loads(manifest_path.read_text())
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    flats=dict(item.split("=",1) for item in args.master_flat)
    if any(c["filter"] not in flats for c in manifest["channels"]):
        raise ValueError("every channel requires an explicitly supplied matching master flat")
    frames=manifest["frames"]
    labels={}
    if args.labels:
        labels={f["path"]:f["userDecision"] for f in json.loads(args.labels.read_text())["frames"]}
    calibration_sources={Path(v): stat_identity(Path(v).stat()) for v in [*flats.values(), str(args.master_dark)]}
    calibration_digests={p.name: 'sha256:'+file_sha256(p) for p in calibration_sources}
    records=[];channels=[]
    with tempfile.TemporaryDirectory(prefix="linear-",dir=output) as temp:
        arrays={}; noises={}; identities={}; native_scales={}
        for f in frames:
            path=Path(f["path"]); identities[f["index"]]=stat_identity(path.stat())
            if 'sha256:'+file_sha256(path)!=f['sourceSha256']:
                raise ValueError(f"source content changed: {path.name}")
            preview=read_frame_preview(path,max_long_edge=2048)
            native_scales[f['index']]=preview.block_size
            geometry=next(c['previewGeometry']['zoom'] for c in manifest['channels'] if c['channelId']==f['channelId'])
            if list(preview.data.shape[::-1])!=geometry:
                raise ValueError('preview geometry differs from the measured registration grid')
            # The production preview helper permits partial calibration. This
            # controlled comparison requires both masters to decode and match.
            for master in (flats[f['filter']], str(args.master_dark)):
                if _master_preview(master, 2048).shape != preview.data.shape:
                    raise ValueError(f"master geometry differs from {path.name}")
            calibrated,ok=calibrate_linear(preview.data,PreviewCalibration(flats[f['filter']],str(args.master_dark),2048))
            if not ok:
                raise ValueError(f"preview calibration unavailable: {path.name}")
            arrays[f['index']]=Path(temp)/f"{f['index']:04d}.npy"
            np.save(arrays[f['index']],calibrated);noises[f['index']]=local_noise(calibrated)
        for c in manifest['channels']:
            members=[f for f in frames if f['channelId']==c['channelId']]
            ref_index=choose_display_reference(members,noises) if args.reference=='local-noise' else c['reference']['index']
            ref_frame=next(f for f in members if f['index']==ref_index)
            ref=DisplayReference.from_image(np.load(arrays[ref_index]))
            ref_matrix=matrix(ref_frame)
            native_scale=native_scales[ref_index]
            points=star_positions(ref)
            legacy_stretch=channel_stretch(ref.image)
            channels.append({'id':c['channelId'],'filter':c['filter'],'referenceIndex':ref_index,'previousReferenceIndex':c['reference']['index'],'referenceName':ref_frame['name'],'noise':ref.noise,'referenceRule':'calibrated-local-noise-psf-v2' if args.reference=='local-noise' else 'existing-v1'})
            for f in members:
                data=np.load(arrays[f['index']]); transformation=compose_to_reference(matrix(f),ref_matrix)
                registered=bool(f['normalization']['registered'] and transformation is not None)
                aligned=warp_to_reference(data,transformation,ref.image.shape) if registered else data
                transparency=f['metrics'].get('transparency');ref_transparency=ref_frame['metrics'].get('transparency')
                gain=ref_transparency/transparency if transparency and ref_transparency else None
                result=diagnostic_preview(aligned,ref,source_noise=noises[f['index']],flux_scale=gain,registered=registered)
                stem=f"{f['index']:04d}"
                # Same calibrated source and same reference in the v1/v2
                # comparison; calibration improvement is a separate column.
                legacy=(aligned-np.median(data))*(gain or 1)+np.median(ref.image)
                legacy_pixels=stretch_to_8bit(legacy,legacy_stretch)
                save_image(output/f'{stem}-v1.png',legacy_pixels/255,(1000,1000))
                save_image(output/f'{stem}-detail.png',result.detail,(1000,1000))
                save_image(output/f'{stem}-field.png',result.field,(1000,1000))
                if result.background_rgb is not None:
                    bg=Image.fromarray(np.rint(result.background_rgb*255).astype(np.uint8)).resize((ref.image.shape[1],ref.image.shape[0]),Image.Resampling.NEAREST)
                    bg.thumbnail((1000,1000));bg.save(output/f'{stem}-background.png')
                old=manifest_path.parent/f['previews']['zoom'] if f['previews'].get('zoom') else None
                if old and old.is_file():
                    with Image.open(old) as im:
                        im.thumbnail((1000,1000));im.save(output/f'{stem}-original.png')
                atlas=native_atlas(f,np.vstack([transformation,[0,0,1]]) if registered else None,points,ref,flats[f['filter']],str(args.master_dark),native_scale) if args.native_crops else None
                if atlas is not None:
                    save_image(output/f'{stem}-native.png',atlas.signal)
                    save_image(output/f'{stem}-shape.png',atlas.shape)
                finite=np.isfinite(aligned)
                records.append({'index':f['index'],'channel':c['channelId'],'filter':f['filter'],'name':f['name'],'night':f['night'],'reference':ref_index,'registered':registered,'human':labels.get(f['name']), 'relativeSignal':1/gain if gain else None,'relativeNoise':result.relative_noise,'matchedSignalNoise':result.matched_signal_noise,'fwhm':f['metrics']['fwhmNative'],'stars':f['metrics']['starCount'],'starRatio':f['metrics']['sourceRatio'],'backgroundSpan':float(np.nanpercentile(result.background_difference,95)-np.nanpercentile(result.background_difference,5))/ref.noise if result.background_difference is not None else None,'v1WhitePercent':float(np.mean(legacy_pixels[finite]>=254)*100),'v2WhitePercent':float(np.mean(result.detail[finite]>=254/255)*100),'stem':stem,'hasBackground':result.background_rgb is not None,'hasNative':atlas is not None,'shapeRegions':atlas.shape_regions if atlas is not None else 0,'hasOriginal':bool(old and old.is_file())})
                print(f"{f['filter']} {f['index']+1}/{len(frames)}",flush=True)
        for f in frames:
            if stat_identity(Path(f['path']).stat())!=identities[f['index']]:
                raise ValueError('source identity changed during study')
    if any(stat_identity(p.stat()) != identity for p,identity in calibration_sources.items()):
        raise ValueError('calibration source changed during study')
    report={'calibrationDigests':calibration_digests,'algorithm':DISPLAY_ALGORITHM,'reference':args.reference,'displayOnly':True,'sourcesVerified':len(frames),'sourcesUnchanged':True,'seconds':time.perf_counter()-start,'channels':channels,'frames':records,'limits':['No human-accuracy claim from display metrics','One mono campaign only','Downsampled views cannot establish native PSF quality','Background differences require registration and measured photometry','Uncalibrated/OSC/native XISF crops are not covered']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    template=Path(__file__).with_name('blink_display_study.html').read_text()
    (output/'index.html').write_text(template.replace('/*REPORT*/',json.dumps(report,ensure_ascii=False).replace('</','<\\/')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--master-flat',action='append',required=True,metavar='FILTER=PATH')
    p.add_argument('--master-dark',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',type=Path)
    p.add_argument('--reference',choices=['existing','local-noise'],default='local-noise')
    p.add_argument('--native-crops',action='store_true')
    args=p.parse_args(); report=run(args)
    print(json.dumps({'frames':report['sourcesVerified'],'seconds':report['seconds'],'output':str(args.output)}))


if __name__=='__main__': main()
