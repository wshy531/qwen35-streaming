"""Exact restoration and bounded rejection of malformed evidence archives."""
import base64,gzip,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'tools/restore_text_bundle.py'
LIMIT=4*2**20

def entry(text):
    data=text.encode('utf-8')
    return dict(text=text,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def envelope(files):
    raw=json.dumps(dict(format='exact_utf8_text_bundle_v1',files=files),ensure_ascii=False).encode('utf-8')
    return dict(encoding='gzip+base64',original_bytes=len(raw),original_sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(gzip.compress(raw)).decode('ascii'))

class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'bundle.json';self.out=self.root/'restored'
    def run_restore(self,value):
        self.source.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
        return subprocess.run([sys.executable,str(SCRIPT),str(self.source),'--out',str(self.out)],capture_output=True,timeout=20)
    def reject(self,value):
        result=self.run_restore(value);self.assertNotEqual(result.returncode,0);self.assertFalse(self.out.exists())
    def test_utf8_original_bytes_and_python_not_executed(self):
        text='# 中文\r\nraise RuntimeError("must not execute")\r\n'
        result=self.run_restore(envelope({'nested/probe.py':entry(text)}));self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.out/'nested/probe.py').read_bytes(),text.encode('utf-8'));self.assertFalse(json.loads(result.stdout)['executed'])
    def test_ten_actual_capsules_restore_exact_bytes(self):
        capsules=sorted((ROOT/'runs').glob('*/full_text.json.gz-base64.json'));self.assertEqual(len(capsules),10)
        for i,source in enumerate(capsules):
            with self.subTest(source=source.parent.name):
                destination=self.root/str(i)
                result=subprocess.run([sys.executable,str(SCRIPT),str(source),'--out',str(destination)],capture_output=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                env=json.loads(source.read_text(encoding='utf-8'));bundle=json.loads(gzip.decompress(base64.b64decode(env['data'])))
                for name,v in bundle['files'].items():self.assertEqual((destination/name).read_bytes(),v['text'].encode('utf-8'))
    def test_path_escape_is_rejected(self):
        for name in ('../escape.py','a/../../escape.py','/absolute.py','C:/escape.py','..\\escape.py'):
            with self.subTest(name=name):self.reject(envelope({name:entry('bad')}))
    def test_late_invalid_file_leaves_no_partial_output(self):
        bad=entry('bad');bad['sha256']='0'*64
        self.reject(envelope({'good.txt':entry('good'),'bad.txt':bad}))
    def test_duplicate_paths_and_file_directory_conflicts(self):
        for names in (('a/./b.py','a/b.py'),('a.py','a.py/b.txt')):
            with self.subTest(names=names):self.reject(envelope({name:entry('text') for name in names}))
    def test_expanded_archive_exceeding_declared_limit_is_rejected(self):
        value=envelope({'probe.py':entry('x')});value['data']=base64.b64encode(gzip.compress(b'x'*(LIMIT+1000))).decode('ascii')
        self.source.write_text(json.dumps(value),encoding='utf-8')
        code="""import gzip,runpy,sys
real=gzip.GzipFile;reads=[]
class Observed(real):
    def read(self,size=-1):
        assert 0<size<=4*2**20+1
        reads.append(size)
        return super().read(size)
gzip.GzipFile=Observed
sys.argv=sys.argv[1:]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except ValueError:pass
else:raise AssertionError('Oversized expansion was accepted')
assert reads==[4*2**20+1],reads
"""
        result=subprocess.run([sys.executable,'-c',code,str(SCRIPT),str(self.source),'--out',str(self.out)],capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertFalse(self.out.exists())
    def test_envelope_size_hash_and_base64_are_checked(self):
        for field,wrong in (('original_bytes',LIMIT+1),('original_sha256','0'*64),('data','not base64!')):
            with self.subTest(field=field):
                value=envelope({'probe.py':entry('x')});value[field]=wrong;self.reject(value)
    def test_existing_output_is_preserved(self):
        self.out.mkdir();sentinel=self.out/'keep.txt';sentinel.write_text('original')
        result=self.run_restore(envelope({'probe.py':entry('new')}));self.assertNotEqual(result.returncode,0);self.assertEqual(sentinel.read_text(),'original')

if __name__=='__main__':unittest.main()
