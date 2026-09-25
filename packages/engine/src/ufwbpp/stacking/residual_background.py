"""Conservative group-relative sky alignment before rejected-sample averaging."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .robust_statistics import nanmedian_rows


@dataclass(frozen=True, slots=True)
class ResidualBackgroundAlignment:
    x_nodes: NDArray[np.float64]
    y_nodes: NDArray[np.float64]
    corrections: NDArray[np.float64]
    evidence: tuple[dict[str, Any], ...]
    weights: tuple[float, ...]

    def apply_coordinates(self, values: NDArray[np.float32], x: NDArray,
                          y: NDArray) -> None:
        for index, grid in enumerate(self.corrections):
            horizontal = np.asarray([np.interp(x, self.x_nodes, row) for row in grid])
            hi = np.clip(np.searchsorted(self.y_nodes, y, side="right"), 1, len(self.y_nodes)-1)
            lo = hi - 1
            wy = np.clip((y-self.y_nodes[lo])/(self.y_nodes[hi]-self.y_nodes[lo]), 0, 1)
            correction = horizontal[lo] * (1-wy[:, None]) + horizontal[hi] * wy[:, None]
            values[index] += correction.astype(np.float32)

    def corrections_at(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float32]:
        """Per-frame Float32 corrections at pixel coordinates ``(x[i], y[i])``.

        Element for element the values ``apply_coordinates`` adds at those
        pixels: the grid rows interpolated along x at the integer columns and
        the same two-node blend along y.
        """
        columns = np.arange(int(np.max(x)) + 1, dtype=np.float64) if x.size else np.zeros(0)
        column_index = x.astype(np.intp)
        hi = np.clip(np.searchsorted(self.y_nodes, y, side="right"), 1, len(self.y_nodes)-1)
        lo = hi - 1
        wy = np.clip((y-self.y_nodes[lo])/(self.y_nodes[hi]-self.y_nodes[lo]), 0, 1)
        result = np.empty((len(self.corrections), x.size), dtype=np.float32)
        for index, grid in enumerate(self.corrections):
            horizontal = np.asarray([np.interp(columns, self.x_nodes, row) for row in grid])
            correction = horizontal[lo, column_index] * (1-wy) + horizontal[hi, column_index] * wy
            result[index] = correction.astype(np.float32)
        return result

    def apply_rows(self, values: NDArray[np.float32], first_row: int) -> None:
        self.apply_coordinates(values, np.arange(values.shape[2], dtype=np.float64),
                               np.arange(first_row, first_row+values.shape[1], dtype=np.float64))

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": "group-temporal-residual-weighted-zero-sky-v1",
            "reference": "registered-group-temporal-median",
            "model": "robust-quadratic-plus-coarse-smoothed-residual",
            "minimumCellSizePixels": 128,
            "residualSmoothingSigmaPixels": 896,
            "weights": list(self.weights),
            "fullStackWeightedMeanConstraint": "zero-weighted-sum-at-every-grid-node",
            "maximumWeightedCorrection": float(np.max(np.abs(np.einsum('i,iyx->yx', self.weights, self.corrections)))),
            "storedInputsModified": False,
            "xNodes": self.x_nodes.tolist(), "yNodes": self.y_nodes.tolist(),
            "frames": [dict(item, correctionGrid=grid.tolist(),
                            correctionSha256="sha256:"+hashlib.sha256(np.asarray(grid,dtype='<f8').tobytes()).hexdigest())
                       for item, grid in zip(self.evidence,self.corrections,strict=True)],
        }


def _robust_fit(nodes: NDArray, valid: NDArray, sigma_nodes: float) -> NDArray:
    """Preserve large-scale gradients at boundaries; smooth only the residual."""
    yy, xx = np.indices(nodes.shape,dtype=np.float64)
    x = 2*xx/max(1,nodes.shape[1]-1)-1
    y = 2*yy/max(1,nodes.shape[0]-1)-1
    design = np.stack([np.ones_like(x),x,y,x*x,x*y,y*y],axis=-1)
    chosen=valid.copy()
    for _ in range(3):
        coefficient=np.linalg.lstsq(design[chosen],nodes[chosen],rcond=None)[0]
        polynomial=np.einsum('yxk,k->yx',design,coefficient)
        residual=nodes-polynomial
        median=float(np.median(residual[chosen]))
        sigma=float(1.4826*np.median(abs(residual[chosen]-median)))
        if sigma<=1e-12: break
        refined=valid & (abs(residual-median)<=3.5*sigma)
        if np.count_nonzero(refined)<12: break
        chosen=refined
    numerator=gaussian_filter(np.where(chosen,residual,0),sigma_nodes,mode='nearest')
    denominator=gaussian_filter(chosen.astype(float),sigma_nodes,mode='nearest')
    return polynomial+numerator/np.maximum(denominator,1e-9)


def fit_residual_background(values: NDArray[np.float32], bin_factor: int,
                            weights: NDArray[np.float64]) -> ResidualBackgroundAlignment | None:
    """Estimate only frame-dependent, coarse residual sky; shared objects cancel.

    No rejected single-reference grid is reused or accepted with relaxed gates.
    Spatial holdout validates each new group-relative model independently.
    """
    n,height,width=values.shape
    cell=max(4,int(np.ceil(128/bin_factor)))
    ys=list(range(0,height,cell));xs=list(range(0,width,cell))
    if n<5 or len(xs)<4 or len(ys)<4:
        return None
    finite=np.isfinite(values)
    common=np.sum(finite,axis=0)>=max(5,(n+1)//2)
    reference=np.full((height,width),np.nan,np.float32)
    reference[common]=np.nanmedian(np.where(finite,values,np.nan)[:,common],axis=0)
    nodes=np.full((n,len(ys),len(xs)),np.nan,np.float64)
    for iy,y0 in enumerate(ys):
        for ix,x0 in enumerate(xs):
            ref=reference[y0:y0+cell,x0:x0+cell]
            valid_ref=np.isfinite(ref)
            if np.count_nonzero(valid_ref)<16: continue
            # The common reference alone selects background positions, avoiding
            # a low-value selection bias in the target frame's noise.
            limit=float(np.quantile(ref[valid_ref],.85))
            background=valid_ref & (ref<=limit)
            # All frames of the cell at once. Unselected samples are NaN, so
            # each row median equals np.median of that frame's compacted
            # selection; the Float32 centre/MAD/clip arithmetic and the
            # 16-sample gates are those of the per-frame loop.
            samples=values[:,y0:y0+cell,x0:x0+cell]-ref
            selected_mask=(background[None,:,:] & np.isfinite(samples)).reshape(n,-1)
            counts=np.count_nonzero(selected_mask,axis=1)
            enough=counts>=16
            if not np.any(enough): continue
            padded=np.where(selected_mask,samples.reshape(n,-1),np.float32(np.nan))
            center=nanmedian_rows(padded)
            deviation=abs(padded-center[:,None])
            sigma=1.4826*nanmedian_rows(deviation)
            clipped=np.where((sigma>0)[:,None],deviation<(3.5*sigma)[:,None],selected_mask)
            final_counts=np.count_nonzero(clipped,axis=1)
            final=nanmedian_rows(np.where(clipped,padded,np.float32(np.nan)))
            fitted=enough & (final_counts>=16)
            nodes[fitted,iy,ix]=final[fitted]
    yy,xx=np.indices(nodes.shape[1:]);checker=(yy+xx)%2==0
    fields=[];evidence=[]
    for frame,observed in enumerate(nodes):
        valid=np.isfinite(observed);training=valid & checker;holdout=valid & ~checker
        if np.mean(valid)<.7 or min(np.sum(training),np.sum(holdout))<8:
            raise ValueError(f"frame {frame}: group residual background is underconstrained")
        sigma_nodes=896/(cell*bin_factor)
        trial=_robust_fit(observed,training,sigma_nodes)
        before=observed[holdout];after=before-trial[holdout]
        # Clip a few defective cells in the independent evaluation, not the
        # target intensities or final science pixels.
        limit=np.quantile(abs(before),.98)
        keep=abs(before)<=limit
        before_rms=float(np.sqrt(np.mean(before[keep]**2)))
        after_rms=float(np.sqrt(np.mean(after[keep]**2)))
        improved=after_rms<.75*before_rms
        field=_robust_fit(observed,valid,sigma_nodes) if improved else np.zeros_like(observed)
        if improved:
            observed_span=float(np.ptp(observed[valid]))
            if np.ptp(field)>1.25*observed_span+6*after_rms:
                raise ValueError(f"frame {frame}: residual background exceeds observed envelope")
        fields.append(field)
        evidence.append({"frameIndex":frame,"status":"APPLIED" if improved else "NOT_BENEFICIAL",
                         "validCellFraction":float(np.mean(valid)),"holdoutCells":int(np.sum(holdout)),
                         "holdoutBeforeRms":before_rms,"holdoutAfterRms":after_rms,
                         "holdoutImprovement":1-after_rms/max(before_rms,1e-12)})
    if not any(item['status']=='APPLIED' for item in evidence): return None
    fields=np.asarray(fields)
    normalized_weights=np.asarray(weights,dtype=np.float64)/np.sum(weights)
    target=np.einsum('i,iyx->yx',normalized_weights,fields)
    corrections=target[None]-fields
    x_nodes=np.array([(start+(min(cell,width-start)-1)/2)*bin_factor+(bin_factor-1)/2 for start in xs])
    y_nodes=np.array([(start+(min(cell,height-start)-1)/2)*bin_factor+(bin_factor-1)/2 for start in ys])
    for array in (x_nodes,y_nodes,corrections):array.setflags(write=False)
    return ResidualBackgroundAlignment(x_nodes,y_nodes,corrections,tuple(evidence),tuple(normalized_weights))
