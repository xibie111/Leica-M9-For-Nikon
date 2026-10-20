import cv2
import numpy as np
from m9.uhdr import SRGB_TO_P3, _seg, icc_p3

def legacy_p3_strip(rgb, x, work, mask):
    """Keep srgb_to_p3's float32 operations, reusing two owned strip buffers."""
    np.clip(rgb, 0, 1, out=x)
    np.less_equal(x, 0.04045, out=mask)
    np.add(x, 0.055, out=work)
    np.divide(work, 1.055, out=work)
    np.power(work, 2.4, out=work)
    np.divide(x, 12.92, out=x)
    np.copyto(work, x, where=mask)
    np.matmul(work, SRGB_TO_P3.T, out=x)
    np.clip(x, 0, 1, out=x)
    np.less_equal(x, 0.0031308, out=mask)
    np.power(x, 1 / 2.4, out=work)
    np.multiply(work, 1.055, out=work)
    np.subtract(work, 0.055, out=work)
    np.multiply(x, 12.92, out=x)
    np.copyto(work, x, where=mask)
    return work

def legacy_encode_jpeg(rgb, quality=95):
    """Use bounded P3 scratch buffers and a contiguous BGR JPEG input."""
    pixels = np.empty(rgb.shape, np.uint8)
    shape = (min(128, rgb.shape[0]), rgb.shape[1], 3)
    x, work = np.empty(shape, np.float32), np.empty(shape, np.float32)
    mask = np.empty(shape, bool)
    for y in range(0, rgb.shape[0], 128):
        part = rgb[y:y + 128]
        stripe = legacy_p3_strip(part, x[:len(part)], work[:len(part)], mask[:len(part)])
        stripe *= 255.0
        stripe += 0.5
        pixels[y:y + 128] = stripe[..., ::-1]
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    data = encoded.tobytes()
    return data[:2] + _seg(0xFFE2, b"ICC_PROFILE\0\x01\x01" + icc_p3()) + data[2:]


"""Standalone regression body, prepended with the literal frozen old encoder."""
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor as StandardPool
import pytest
from m9 import nef


def sample(height=1025, width=31):
    return np.random.default_rng(20261024).uniform(-.2, 1.2, (height, width, 3)).astype(np.float32)


def outcome(fn, rgb, quality=95):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            value=fn(rgb, quality);result=('ok',value)
        except Exception as error:
            result=('error',type(error),str(error))
    return result,[(v.category,str(v.message)) for v in caught]


def pool_probe(monkeypatch, barrier=False):
    rows=[]
    class Pool(StandardPool):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.row=dict(max_workers=self._max_workers,tasks=[],closed=False);rows.append(self.row)
            self.gate=threading.Barrier(4) if barrier else None
        def submit(self,fn,*a,**kw):
            row=dict(args=a);self.row['tasks'].append(row)
            def run():
                row.update(thread=threading.get_native_id(),name=threading.current_thread().name)
                if self.gate:self.gate.wait(5)
                try:return fn(*a,**kw)
                finally:row['finished']=True
            return super().submit(run)
        def shutdown(self,*a,**kw):
            try:return super().shutdown(*a,**kw)
            finally:self.row.update(closed=True,alive=[t.name for t in self._threads if t.is_alive()])
    monkeypatch.setattr(nef,'ThreadPoolExecutor',Pool)
    return rows


def assert_pool(rows, count=1):
    assert len(rows)==count
    for row in rows:
        assert row['max_workers']==4 and row['closed'] and not row['alive']
        assert len(row['tasks'])==4 and all(t['finished'] for t in row['tasks'])
        assert [t['args'] for t in row['tasks']]==[(0,),(128,),(256,),(384,)]


@pytest.mark.parametrize('height',[127,128,129,511,512,513,1025])
def test_grid_boundaries_full_bytes_bgr_and_single_encoder(monkeypatch,height):
    rgb=sample(height);before=rgb.view(np.uint32).copy();rows=pool_probe(monkeypatch)
    original=cv2.imencode;captured=[]
    def encode(ext,pixels,params):
        assert not [t for t in threading.enumerate() if t.name.startswith('m9-p3')]
        captured.append((ext,pixels.copy(),params,threading.get_ident()))
        return original(ext,pixels,params)
    monkeypatch.setattr(cv2,'imencode',encode)
    expected=legacy_encode_jpeg(rgb);actual=nef._encode_jpeg(rgb)
    assert actual==expected and len(captured)==2
    assert captured[0][0]==captured[1][0]=='.jpg' and captured[0][2]==captured[1][2]==[cv2.IMWRITE_JPEG_QUALITY,95]
    assert captured[0][3]==captured[1][3]==threading.get_ident()
    assert captured[1][1].flags.c_contiguous and np.array_equal(captured[0][1],captured[1][1])
    assert np.array_equal(rgb.view(np.uint32),before)
    if height>=512:assert_pool(rows)
    else:assert not rows


def test_four_real_threads_independent_scratch_bits_and_exact_row_coverage(monkeypatch):
    import hashlib,sys
    rgb=sample(1025,37);rgb.flags.writeable=False;before=rgb.tobytes();rows=pool_probe(monkeypatch,True)
    code=nef._p3_strip.__code__;records=[];lock=threading.Lock();base=rgb.ctypes.data
    first=threading.Barrier(4);seen=set();original=np.matmul
    def matmul(*a,**kw):
        ident=threading.get_native_id()
        with lock:fresh=ident not in seen;seen.add(ident)
        if fresh and threading.current_thread().name.startswith('m9-p3'):first.wait(5)
        return original(*a,**kw)
    monkeypatch.setattr(np,'matmul',matmul)
    def profile(frame,event,value):
        if frame.f_code is code and event=='return' and isinstance(value,np.ndarray):
            local=frame.f_locals;part=local['rgb'];shape=part.shape
            x=np.empty(shape,np.float32);work=np.empty(shape,np.float32);mask=np.empty(shape,bool)
            expected=legacy_p3_strip(part,x,work,mask)
            assert np.array_equal(value.view(np.uint32),expected.view(np.uint32))
            row=dict(y=(part.ctypes.data-base)//rgb.strides[0],shape=shape,thread=threading.get_native_id(),
                scratch=tuple(local[k].ctypes.data for k in ['x','work','mask']),
                strides=part.strides,sha=hashlib.sha256(value.tobytes()).hexdigest())
            with lock:records.append(row)
    old=sys.getprofile();old_thread=threading.getprofile()
    try:
        sys.setprofile(profile);threading.setprofile(profile)
        actual=nef._encode_jpeg(rgb)
    finally:sys.setprofile(old);threading.setprofile(old_thread)
    assert actual==legacy_encode_jpeg(rgb) and rgb.tobytes()==before and not rgb.flags.writeable
    assert_pool(rows);assert len({t['thread'] for t in rows[0]['tasks']})==4
    assert sorted(v['y'] for v in records)==list(range(0,1025,128))
    owners={}
    for r in records:
        assert r['shape']==(min(128,1025-r['y']),37,3) and r['strides']==rgb.strides
        if r['thread'] in owners:assert owners[r['thread']]==r['scratch']
        owners[r['thread']]=r['scratch']
    assert len(owners)==4 and len({p for v in owners.values() for p in v})==12


@pytest.mark.parametrize('layout',['F64','int16','F','slice','negative','readonly','subclass','bigendian','nan','snan','inf','empty','twoD','channels','scalar'])
def test_generic_fallback_preserves_old_bytes_warnings_errors_and_input(monkeypatch,layout):
    rgb=sample()
    class Subclass(np.ndarray):pass
    if layout=='F64':rgb=rgb.astype(np.float64)
    elif layout=='int16':rgb=(rgb*255).astype(np.int16)
    elif layout=='F':rgb=np.asfortranarray(rgb)
    elif layout=='slice':rgb=rgb[::2,::2]
    elif layout=='negative':rgb=rgb[::-1]
    elif layout=='readonly':rgb.flags.writeable=False
    elif layout=='subclass':rgb=rgb.view(Subclass)
    elif layout=='bigendian':rgb=rgb.astype('>f4')
    elif layout=='nan':rgb[0,0,0]=np.nan
    elif layout=='snan':rgb.view(np.uint32)[0,0,0]=0x7f800001
    elif layout=='inf':rgb[0,0,0]=np.inf
    elif layout=='empty':rgb=rgb[:0]
    elif layout=='twoD':rgb=rgb[...,0]
    elif layout=='channels':rgb=rgb[...,:2]
    elif layout=='scalar':rgb=np.array(1,np.float32)
    before=(rgb.tobytes(),rgb.shape,rgb.strides,rgb.flags.writeable)
    rows=pool_probe(monkeypatch)
    assert outcome(nef._encode_jpeg,rgb)==outcome(legacy_encode_jpeg,rgb)
    assert before==(rgb.tobytes(),rgb.shape,rgb.strides,rgb.flags.writeable)
    if layout=='readonly':assert_pool(rows)
    else:assert not rows


@pytest.mark.parametrize('quality',[0,90,100,-1,101,np.int64(95),True,95.0,'95',None,[95]])
def test_other_quality_keeps_old_cv2_behavior_and_serial_order(monkeypatch,quality):
    rgb=sample();rows=pool_probe(monkeypatch)
    assert outcome(nef._encode_jpeg,rgb,quality)==outcome(legacy_encode_jpeg,rgb,quality)
    assert not rows


@pytest.mark.parametrize('policy',['raise','call','log','print','ignore','under_raise','callback_only'])
def test_caller_errorstate_callback_thread_and_first_exception_are_preserved(monkeypatch,capfd,policy):
    rgb=sample();rows=pool_probe(monkeypatch);calls=[]
    if policy=='under_raise':rgb.fill(np.nextafter(np.float32(0),np.float32(1)))
    elif policy!='callback_only':rgb.view(np.uint32)[0,0,0]=0x7f800001
    class Log:
        def write(self,text):calls.append(('log',text,threading.get_ident()))
    def handler(message,flag):calls.append(('call',message,flag,threading.get_ident()))
    saved=np.geterrcall();settings={'under':'raise'} if policy=='under_raise' else ({} if policy=='callback_only' else {'all':policy})
    def invoke(fn):
        calls.clear();np.seterrcall(Log() if policy=='log' else handler)
        with np.errstate(**settings):result=outcome(fn,rgb)
        return result,list(calls),capfd.readouterr()
    try:
        expected=invoke(legacy_encode_jpeg);actual=invoke(nef._encode_jpeg)
        assert actual==expected and not rows
        assert all(v[-1]==threading.get_ident() for v in actual[1])
    finally:np.seterrcall(saved)


def test_private_strip_hook_keeps_main_thread_sequential_calls(monkeypatch):
    rgb=sample();rows=pool_probe(monkeypatch);calls=[]
    def hook(part,*buffers):
        calls.append(((part.ctypes.data-rgb.ctypes.data)//rgb.strides[0],threading.get_ident()))
        return np.clip(part,0,1).astype(np.float32)
    monkeypatch.setattr(nef,'_p3_strip',hook);monkeypatch.setitem(legacy_encode_jpeg.__globals__,'legacy_p3_strip',hook)
    expected=legacy_encode_jpeg(rgb);first=list(calls);calls.clear()
    assert nef._encode_jpeg(rgb)==expected and calls==first==[(y,threading.get_ident()) for y in range(0,1025,128)] and not rows


def test_worker_math_exception_propagates_after_all_threads_join(monkeypatch):
    rgb=sample();rows=pool_probe(monkeypatch,True);gate=threading.Barrier(4);seen=set();lock=threading.Lock();original=np.matmul
    def matmul(*a,**kw):
        ident=threading.get_ident()
        with lock:fresh=ident not in seen;seen.add(ident)
        if fresh:
            if gate.wait(5)==0:raise RuntimeError('p3-worker-fault')
        return original(*a,**kw)
    monkeypatch.setattr(np,'matmul',matmul)
    monkeypatch.setattr(cv2,'imencode',lambda *a,**kw:pytest.fail('encoding after worker failure'))
    with pytest.raises(RuntimeError,match='p3-worker-fault'):nef._encode_jpeg(rgb)
    assert_pool(rows)
    assert len(seen)==4 and not [t for t in threading.enumerate() if t.name.startswith('m9-p3')]


@pytest.mark.parametrize('failure',['false','raise'])
def test_encoder_error_occurs_on_caller_after_pool_join(monkeypatch,failure):
    rgb=sample();rows=pool_probe(monkeypatch);calls=[]
    def encode(*a,**kw):
        assert not [t for t in threading.enumerate() if t.name.startswith('m9-p3')]
        calls.append(threading.get_ident())
        if failure=='raise':raise ValueError('encode-fault')
        return False,None
    monkeypatch.setattr(cv2,'imencode',encode)
    assert outcome(nef._encode_jpeg,rgb)==outcome(legacy_encode_jpeg,rgb)
    assert calls==[threading.get_ident()]*2;assert_pool(rows)


def test_submit_error_still_joins_started_worker(monkeypatch):
    rgb=sample();rows=[]
    class Pool(StandardPool):
        def submit(self,*a,**kw):
            if self._threads:raise RuntimeError('submit-fault')
            return super().submit(*a,**kw)
        def shutdown(self,*a,**kw):
            try:return super().shutdown(*a,**kw)
            finally:rows.append([t.is_alive() for t in self._threads])
    monkeypatch.setattr(nef,'ThreadPoolExecutor',Pool)
    with pytest.raises(RuntimeError,match='submit-fault'):nef._encode_jpeg(rgb)
    assert rows and not any(rows[0]) and not [t for t in threading.enumerate() if t.name.startswith('m9-p3')]


def test_threshold_and_u8_rounding_neighbors_keep_p3_uint32_and_parallel_bytes(monkeypatch):
    values=np.array([-.2,-0.,0.,.04045,.0031308,1.,1.2,*[(i+.5)/255 for i in range(256)]],np.float32)
    values=np.concatenate((values,np.nextafter(values,np.float32(-np.inf)),np.nextafter(values,np.float32(np.inf))))
    rgb=np.resize(values,(1025,37,3));before=rgb.view(np.uint32).copy();rows=pool_probe(monkeypatch)
    for y in range(0,len(rgb),128):
        part=rgb[y:y+128];shape=part.shape
        a=nef._p3_strip(part,np.empty(shape,np.float32),np.empty(shape,np.float32),np.empty(shape,bool))
        b=legacy_p3_strip(part,np.empty(shape,np.float32),np.empty(shape,np.float32),np.empty(shape,bool))
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
    assert nef._encode_jpeg(rgb)==legacy_encode_jpeg(rgb);assert_pool(rows)
    assert np.array_equal(rgb.view(np.uint32),before)
