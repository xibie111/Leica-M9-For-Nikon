"""Frozen legacy TDL oracle and public/internal boundary regression contracts."""
import hashlib
import sys
import warnings
import numpy as np
import pytest
from m9 import frontend as f


def legacy_tdl(rgb: np.ndarray, hue_tab: np.ndarray, sat_tab: np.ndarray) -> np.ndarray:
    """2D LUT: HSV hue/sat grid, 24 hue knots (15 deg) x 16 sat knots."""
    import cv2
    x = np.clip(rgb, 0, None).astype(np.float32)
    v = x.max(-1)
    scale = np.maximum(v, 1e-6)
    hsv = cv2.cvtColor(x / scale[..., None], cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    hi = h / 15.0
    si = s * 15.0
    h0 = np.floor(hi).astype(int) % 24
    h1 = (h0 + 1) % 24
    fh = hi - np.floor(hi)
    s0 = np.clip(np.floor(si).astype(int), 0, 15)
    s1 = np.clip(s0 + 1, 0, 15)
    fs = np.clip(si - s0, 0, 1)

    def bil(t):
        return ((1 - fh) * ((1 - fs) * t[h0, s0] + fs * t[h0, s1])
                + fh * ((1 - fs) * t[h1, s0] + fs * t[h1, s1]))
    hsv[..., 0] = np.mod(h + bil(hue_tab), 360.0)
    hsv[..., 1] = np.clip(s * (1.0 + bil(sat_tab)), 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * scale[..., None]
    return out


def tables():
    return f.tuning.tdl_tables('tele',flag=0.0,drc=1.4,lux=300.,cct=5100.)


def outcome(callback,*args):
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        try:value=callback(*args);error=None
        except Exception as e:value=None;error=(type(e),str(e))
    return value,error,[(w.category.__name__,str(w.message)) for w in messages]


def assert_outcome(a,b):
    assert a[1:]==b[1:]
    if a[1] is not None:return
    left,right=a[0],b[0]
    assert type(left) is type(right) and left.dtype==right.dtype and left.shape==right.shape
    assert left.strides==right.strides and left.flags.owndata==right.flags.owndata
    assert np.array_equal(left.view(np.uint32),right.view(np.uint32))


def digest(x):return hashlib.sha256(x.tobytes(order='A')).hexdigest()


@pytest.mark.parametrize('layout',['C','F','slice','negative_stride','readonly'])
@pytest.mark.parametrize('special',[False,True])
def test_prepared_bits_layout_ownership_and_input(layout,special):
    rgb=np.random.default_rng(20261023).uniform(0,3.5,(19,47,3)).astype(np.float32)
    # RGB primaries/greys on both sides of scale epsilon, hue/saturation knots,
    # values above one and signed zeros; no replacement HSV arithmetic.
    rgb.reshape(-1,3)[:8]=[[0.,-0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
        [1e-6,1e-6,1e-6],[np.nextafter(np.float32(1e-6),np.float32(0)),0.,0.],
        [2.,1.,.5],[3.,3.,3.]]
    if special:
        rgb.reshape(-1)[:10]=np.array([0x00000000,0x80000000,0x00000001,0x3f800001,0x7f800000,0xff800000,0x7fc01234,0xffc04321,0x7fa01234,0x3f7fffff],np.uint32).view(np.float32)
    with np.errstate(invalid='ignore'):rgb=np.clip(rgb,0,None)
    if layout=='F':rgb=np.asfortranarray(rgb)
    elif layout=='slice':rgb=rgb[:,::2,:]
    elif layout=='negative_stride':rgb=rgb[::-1,::-1,:]
    elif layout=='readonly':rgb.flags.writeable=False
    before=digest(rgb);shape=rgb.shape;strides=rgb.strides;writable=rgb.flags.writeable
    a=outcome(legacy_tdl,rgb,*tables());b=outcome(f._tdl_apply_prepared,rgb,*tables())
    assert_outcome(a,b)
    assert digest(rgb)==before and rgb.shape==shape and rgb.strides==strides and rgb.flags.writeable==writable
    if b[1] is None:
        assert b[0].dtype==np.float32 and b[0].flags.owndata and b[0].flags.c_contiguous and not np.shares_memory(rgb,b[0])
        if not special:assert np.max(b[0])>1.,'TDL must retain values above one'


@pytest.mark.parametrize('kind',['float64','int16','negative','subclass','F','slice','readonly','malformed'])
def test_public_keeps_legacy_generic_behavior(kind):
    rgb=np.random.default_rng(31).uniform(-.3,2.,(11,23,3)).astype(np.float32)
    if kind=='float64':rgb=rgb.astype(np.float64)
    elif kind=='int16':rgb=(rgb*11).astype(np.int16)
    elif kind=='subclass':
        class RGB(np.ndarray):pass
        rgb=rgb.view(RGB)
    elif kind=='F':rgb=np.asfortranarray(rgb)
    elif kind=='slice':rgb=rgb[:,::-2,:]
    elif kind=='readonly':rgb.flags.writeable=False
    elif kind=='malformed':rgb=rgb[...,0]
    before=digest(rgb);shape=rgb.shape;strides=rgb.strides;writable=rgb.flags.writeable
    assert_outcome(outcome(legacy_tdl,rgb,*tables()),outcome(f.tdl_apply,rgb,*tables()))
    assert digest(rgb)==before and rgb.shape==shape and rgb.strides==strides and rgb.flags.writeable==writable


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_real_internal_entry_and_generic_fallback(monkeypatch,dtype):
    lin=np.random.default_rng(71).uniform(.001,1.2,(67,99,3)).astype(dtype)
    matrix=np.eye(3,dtype=np.float32);info=f.RawInfo(cct=5100.,ev100=11.)
    before=digest(lin)
    with monkeypatch.context() as patch:
        patch.setattr(f,'tdl_apply',legacy_tdl)
        expected=f.b2y_from_raw(lin,matrix,info,camera='tele')
    calls=[];original=f._tdl_apply_prepared;public_code=f.tdl_apply.__code__
    def core(x,*args):
        calls.append(dict(from_public=sys._getframe(1).f_code is public_code,dtype=x.dtype,shape=x.shape))
        return original(x,*args)
    monkeypatch.setattr(f,'_tdl_apply_prepared',core)
    actual=f.b2y_from_raw(lin,matrix,info,camera='tele')
    assert calls and all(row['from_public']==(dtype==np.float64) for row in calls)
    assert np.array_equal(actual.rgb.view(np.uint32),expected.rgb.view(np.uint32))
    assert actual.info is info and actual.lin is lin and actual.cam2srgb is matrix and digest(lin)==before
