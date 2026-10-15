"""Small CPU synthetic optimizer smoke; never loads real datasets/CLIP or CUDA."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import copy
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from t2methods.models import build_model
from t2methods.checkpoint import load_checkpoint, model_state_hash, completed_run, file_hash, archive_failed_attempt, write_json
from t2methods.training import Trainer, seed_everything
from t2methods.evaluation import evaluate_protocol
from tests.fake_clip import FakeClip, CLIP_MODULE, config as model_config

torch.set_num_threads(1)

class IntegrationChecks(unittest.TestCase):
    def fixture(self, root, method):
        root.mkdir(); classes=["class0","class1","class2"]
        taxonomy={"classes":{name:{"leaf":name,"template":"a photo of land "+name+"."} for name in classes}}
        taxonomy_path=root/"taxonomy.json"; taxonomy_path.write_text(json.dumps(taxonomy),encoding="utf-8")
        records=[]
        for label in range(3):
            for index in range(3):
                array=np.zeros((24,24,3),dtype=np.uint8); array[...,label]=70+index*40
                array[:,:, (label+1)%3]=np.arange(24,dtype=np.uint8)[None,:]*6
                path=root/('%s_%d.png'%(classes[label],index)); Image.fromarray(array).save(path)
                records.append({"id":path.name,"path":str(path),"label":label,"classname":classes[label],"split":"train" if index<2 else "test"})
        support=[r for r in records if r["split"]=="train"]; test=[r for r in records if r["split"]=="test"]
        bundle=SimpleNamespace(train=support,val=[],train_classnames=classes,test=test,test_classnames=classes,
                               base_test=test[:2],base_classnames=classes[:2],new_test=[dict(test[2],label=0)],new_classnames=classes[2:],
                               metadata={"role":"source","support_ids":[r["id"] for r in support],"support_sha256":"synthetic_support",
                                         "support_record_sha256":"synthetic_record","split_sha256":"synthetic_split","seed":1,"shot":2})
        config=model_config(method); config.update(data={"taxonomy":str(taxonomy_path),"task":"b2n","dataset":"SyntheticCPU"},
            train={"device":"cpu","precision":"fp32","epochs":2,"workers":0,"batch_size":3,"eval_batch_size":3,
                   "learning_rate":.0025,"weight_decay":.00001,"warmup_epochs":1,"warmup_learning_rate":.00001,
                   "crop_min":.08,"horizontal_flip":.5,"grad_scaler_initial_scale":128,"max_minutes_per_run":2},
            runtime={"phase":"formal", "save_scores":True, "save_features":True, "diagnostic_gradient_batches":1},_seed=1,_request_identity={"fixture":"engineering CPU synthetic only","method":method},
            _clip_source_hashes={"fake":"not_original_clip"},_clip_checkpoint_sha256="synthetic_only",
            _eval_preprocess=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))]))
        return bundle,config

    def model(self, method, config):
        seed_everything(123,"cpu")
        return build_model(method,FakeClip(input_size=224),CLIP_MODULE,config)

    def test_spcp_cpu_optimizer_checkpoint_resume_and_transfer(self):
        scratch=Path(os.path.abspath(__file__)).parents[1]/"test_scratch"; scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch)) as directory:
            Path(os.path.abspath(directory)).relative_to(scratch)
            root=Path(directory)
            for method in ["CPTA"]:
                with self.subTest(method=method):
                    bundle,config=self.fixture(root/method,method)
                    full=root/method/"full"; partial=root/method/"partial"; full.mkdir(); partial.mkdir()
                    full_model=self.model(method,config); result=Trainer(full_model,bundle,config,full).run()
                    self.assertEqual(result["completed_epochs"],2)
                    self.assertTrue(completed_run(full,config["_request_identity"]))
                    self.assertTrue(json.loads((full/"audit.json").read_text())["backbone_unchanged"])
                    self.assertTrue((full/"base_predictions_scores.npz").is_file())
                    self.assertTrue((full/"mechanism_diagnostics.jsonl").is_file())
                    import t2methods.training as training
                    original_save=training.save_checkpoint
                    def save_and_interrupt(path,state):
                        original_save(path,state)
                        if state["completed_epochs"]==1: raise RuntimeError("intentional CPU smoke epoch boundary interruption")
                    with patch.object(training,"save_checkpoint",save_and_interrupt):
                        with self.assertRaisesRegex(RuntimeError,"intentional CPU smoke"):
                            Trainer(self.model(method,config),bundle,config,partial).run()
                    saved=load_checkpoint(partial/"last.pt"); self.assertEqual(saved["completed_epochs"],1)
                    resumed_model=self.model(method,config); resumed_trainer=Trainer(resumed_model,bundle,config,partial)
                    with patch.object(resumed_model,"prepare",side_effect=AssertionError("resume must not recalibrate")):
                        resumed=resumed_trainer.run(resume=partial/"last.pt")
                    self.assertEqual(result["state_sha256"],resumed["state_sha256"])
                    self.assertEqual(result["hm"],resumed["hm"])
                    for name in ["base_predictions.csv","new_predictions.csv"]:
                        self.assertEqual(file_hash(full/name),file_hash(partial/name))
                    replay=root/method/"replay"; replay.mkdir()
                    restored=self.model(method,config); trainer=Trainer(restored,bundle,config,replay)
                    with patch.object(restored,"prepare",side_effect=AssertionError("evaluation must not recalibrate")):
                        replay_result=trainer.evaluate_checkpoint(full/"final.pt")
                    self.assertEqual(result["state_sha256"],replay_result["state_sha256"])
                    failed=root/method/"failed_evaluation"; failed.mkdir()
                    for name in ["final.pt","audit.json","metadata.json"]: shutil.copyfile(full/name,failed/name)
                    write_json(failed/"status.json",{"status":"failed"}); write_json(failed/"failure.json",{"error":"synthetic evaluation interruption"})
                    (failed/"base_predictions.csv").write_text("partial",encoding="utf-8")
                    archive_failed_attempt(failed,"evaluate_final")
                    recovery_model=self.model(method,config); recovery=Trainer(recovery_model,bundle,config,failed)
                    with patch.object(recovery_model,"prepare",side_effect=AssertionError("evaluation recovery must not prepare")),patch.object(recovery,"run",side_effect=AssertionError("evaluation recovery must not train")):
                        recovered=recovery.evaluate_checkpoint(failed/"final.pt",recover_final=True)
                    self.assertEqual(result["state_sha256"],recovered["state_sha256"])
                    self.assertTrue(completed_run(failed,config["_request_identity"]))
                    state_before=model_state_hash(restored)
                    for task in ["cd","dg"]:
                        out=root/method/task; out.mkdir(); transfer=copy.deepcopy(config); transfer["data"]["task"]=task
                        targets=[]
                        for i in range(3):
                            target=SimpleNamespace(train=[],val=[],test=bundle.new_test,test_classnames=bundle.new_classnames,
                                                   metadata={"role":"target","support_ids":[]})
                            targets.append(({"dataset":"SyntheticTarget%d"%i,"taxonomy":config["data"]["taxonomy"]},target))
                        evaluated=evaluate_protocol(restored,bundle,targets,config["_eval_preprocess"],transfer,out)
                        self.assertEqual(len(evaluated["targets"]),3)
                        self.assertEqual(state_before,model_state_hash(restored))
                    # A forged successful marker cannot skip a modified artifact.
                    (full/"base_predictions.csv").write_text("tampered",encoding="utf-8")
                    self.assertFalse(completed_run(full,config["_request_identity"]))

if __name__=="__main__": unittest.main()
