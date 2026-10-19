"""Exercise malformed archives through the real restore command, before writes."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CAP=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()


class RestoreBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.cap=self.root/'capsule';(self.cap/'tools').mkdir(parents=True)
        (self.cap/'archive').mkdir();self.out=self.root/'restored'
        for name in ['verify.py','restore_workspace.py']:shutil.copyfile(CAP/'tools'/name,self.cap/'tools'/name)

    def capsule(self,name='nested/probe.py',text='# 中文\r\nraise RuntimeError("never execute")\r\n',growth=False,bad_sha=False):
        data=text.encode('utf-8');entry=dict(text=text,bytes=len(data),sha256=sha(data))
        raw=json.dumps(dict(format='exact_utf8_text_bundle_v1',files={name:entry}),ensure_ascii=False).encode('utf-8')
        if growth:raw=b'x'*(4*2**20+1)
        envelope=dict(encoding='gzip+base64',original_bytes=1 if growth else len(raw),original_sha256='0'*64 if bad_sha else sha(raw),data=base64.b64encode(gzip.compress(raw)).decode())
        encoded=json.dumps(envelope).encode();(self.cap/'archive/one.json').write_bytes(encoded)
        index=dict(format='bounded_exact_text_collection_v1',bundles=[dict(path='archive/one.json',bytes=len(encoded),sha256=sha(encoded),files=1,original_bytes=len(raw))],files={name:dict(bytes=len(data),sha256=sha(data))})
        (self.cap/'ARCHIVE_INDEX.json').write_text(json.dumps(index),encoding='utf-8');(self.cap/'SOURCE_MAP.json').write_text('{}',encoding='utf-8')
        return data

    def run_restore(self,optimized=False):
        return subprocess.run([sys.executable,*(['-O'] if optimized else []),str(self.cap/'tools/restore_workspace.py'),'--out',str(self.out)],capture_output=True,timeout=15)

    def test_exact_utf8_and_no_execution(self):
        data=self.capsule();r=self.run_restore();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual((self.out/'nested/probe.py').read_bytes(),data);self.assertFalse(json.loads(r.stdout)['executed'])

    def test_escape_refused_before_creation(self):
        self.capsule(name='../escape.py');self.assertNotEqual(self.run_restore().returncode,0)
        self.assertFalse(self.out.exists());self.assertFalse((self.root/'escape.py').exists())

    def test_identity_refused_before_creation(self):
        self.capsule(bad_sha=True);self.assertNotEqual(self.run_restore().returncode,0);self.assertFalse(self.out.exists())

    def test_expansion_limit_refused_before_creation(self):
        self.capsule(growth=True);self.assertNotEqual(self.run_restore().returncode,0);self.assertFalse(self.out.exists())

    def test_existing_output_untouched(self):
        self.capsule();self.out.mkdir();(self.out/'marker').write_bytes(b'keep')
        self.assertNotEqual(self.run_restore().returncode,0);self.assertEqual((self.out/'marker').read_bytes(),b'keep')

    def test_optimization_cannot_disable_checks(self):
        self.capsule();self.assertNotEqual(self.run_restore(optimized=True).returncode,0);self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main(verbosity=2)
