import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deploy_edge',ROOT/'tools/deploy_edge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeployTests(unittest.TestCase):
    def make_repo(self, root):
        repo = root/'repo'
        for name,text in [('edge/darknet53/live.py','new'),
                          ('edge/fcm_tools/verify_live_fcm.py','verify'),
                          ('edge/network_test/board_send.py','transport')]:
            p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        return repo

    def test_dry_run_does_not_create_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=self.make_repo(root);target=root/'target'
            module.deploy(repo,target)
            self.assertFalse(target.exists())

    def test_existing_script_backed_up_and_models_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=self.make_repo(root);target=root/'target'
            old=target/'darknet53/live.py';old.parent.mkdir(parents=True);old.write_text('old')
            model=old.parent/'dark3_fp.rknn';model.write_bytes(b'existing-model')
            result=module.deploy(repo,target,True)
            self.assertEqual(old.read_text(),'new')
            self.assertEqual((Path(result['backup'])/'darknet53/live.py').read_text(),'old')
            self.assertEqual(model.read_bytes(),b'existing-model')
            self.assertEqual((target/'fcm_l37/verify_live_fcm.py').read_text(),'verify')
            self.assertEqual((target/'network_test/board_send.py').read_text(),'transport')
            self.assertEqual(module.deploy(repo,target,True)['changed'],0)

    def test_parent_symlink_cannot_redirect_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=self.make_repo(root);target=root/'target'
            target.mkdir();outside=root/'outside';outside.mkdir()
            (target/'darknet53').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):
                module.deploy(repo,target,True)
            self.assertEqual(list(outside.iterdir()),[])


if __name__=='__main__':
    unittest.main()
