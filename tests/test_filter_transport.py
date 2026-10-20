"""Small exact transport contracts; full native CUDA sizes live in frozen evidence."""
import numpy as np
import pytest
import torch
from m9 import leicafilter as f

@pytest.mark.parametrize('shape',[(101,23,3),(77,37,3),(23,77,3)])
@pytest.mark.parametrize('style',[True,False])
def test_independent_grids_and_exact_owned_output(monkeypatch,shape,style):
    monkeypatch.setattr(f,'_LUT_CHUNK_PIXELS',1000)
    monkeypatch.setattr(f,'_VIGNETTE_CHUNK_PIXELS',333)
    rgb=np.random.default_rng(9).random(shape,dtype=np.float32)
    before=rgb.copy();params=f.LeicaFilter(device='cpu')
    p=params.snapshot if style else params.preview
    lut=p.lut('common',200,5100);shade=p.shading_params(3.2,200,5100)
    # Same row grids explicitly supplied to the legacy oracle.
    original_lut=f._lut_apply_torch
    mapped=np.empty_like(rgb);step=max(1,1000//shape[1])
    for y in range(0,shape[0],step):mapped[y:y+step]=original_lut(torch,'cpu',rgb[y:y+step],lut)
    expected=f._cvstyle_torch(torch,'cpu',mapped,shade,7.5,chunk_rows=max(1,333//shape[1]))
    trace=[];actual=f._lut_cvstyle_torch(torch,'cpu',rgb,lut,shade,7.5,trace=lambda *v:trace.append(v))
    assert np.array_equal(expected.view(np.uint32),actual.view(np.uint32))
    assert np.array_equal(before.view(np.uint32),rgb.view(np.uint32))
    assert actual.dtype==np.float32 and actual.flags.owndata and actual.flags.c_contiguous
    for stage,pixels in [('lut',1000),('vignette',333)]:
        step=max(1,pixels//shape[1])
        assert [(y,rows,h) for name,y,rows,h in trace if name==stage]==[(y,min(step,shape[0]-y),shape[0]) for y in range(0,shape[0],step)]

@pytest.mark.parametrize('style',[True,False])
def test_cpu_and_shading_off_keep_public_routes(monkeypatch,style):
    image=np.ones((7,11,3),np.float32)*.5;calls=[]
    original_lut=f.lut_apply;original_vig=f.cvstyle_apply
    def lut(*a,**k):calls.append('lut');return original_lut(*a,**k)
    def vig(*a,**k):calls.append('vignette');return original_vig(*a,**k)
    monkeypatch.setattr(f,'lut_apply',lut);monkeypatch.setattr(f,'cvstyle_apply',vig)
    obj=f.LeicaFilter(device='cpu');obj(image,200,5100,style_trans_on=style)
    assert calls==['lut','vignette'];calls.clear()
    obj(image,200,5100,style_trans_on=style,shading=False)
    assert calls==['lut']
