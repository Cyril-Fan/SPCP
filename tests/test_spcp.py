"""CPU implementation checks for full SPCP, independent of benchmark images."""
import copy
from types import SimpleNamespace
import unittest
import torch
from tests.fake_clip import FakeClip,CLIP_MODULE,config,vocabulary
from spcp.models import SPCPModel
from t2methods.models.cpta import edge_loss
from t2methods.models import build_model
from t2methods.models.source_graph import neighbors
from t2methods.checkpoint import model_state_hash

torch.set_num_threads(1)

def model():
    torch.manual_seed(123)
    m=SPCPModel(FakeClip(),CLIP_MODULE,config('CPTA',eta=.5))
    vocabulary(m)
    m.prepare(SimpleNamespace(train=[{'id':'synthetic'}],metadata={'role':'source','support_sha256':'synthetic'}),None,1,None)
    return m

class SPCPChecks(unittest.TestCase):
    def test_joint_loss_and_frozen_backbone(self):
        m=model();images=torch.linspace(-1,1,2*3*8*8).reshape(2,3,8,8)
        terms=m.loss_terms(images,torch.tensor([0,1]))
        expected=terms['ce']+terms['image_alignment']+.5*terms['text_alignment']+.5*terms['mechanism_loss']
        torch.testing.assert_close(terms['loss'],expected)
        terms['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.clip_model.parameters()))
        self.assertTrue(all(not b.requires_grad for b in m._method_buffers().values()))

    def test_same_augmented_image_teacher(self):
        m=model();images=torch.randn(2,3,8,8);out=m(images)
        teacher=m.teacher_target(images,out)
        torch.testing.assert_close(teacher,m.reference_image(images))
        self.assertFalse(teacher.requires_grad)

    def test_full_coordinate_edges_zero_on_reference(self):
        m=model()
        value,_,_=edge_loss(m.source_q,m.source_q,m.source_neighbors,m.source_neighbor_mask,m.source_delta,m.source_denominator)
        self.assertEqual(float(value),0.)
        self.assertTrue(torch.all(m.source_denominator>=.2**2))

    def test_duplicate_filter_and_deterministic_ties(self):
        q=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,-1.]])
        index,mask=neighbors(q,3,exclude_duplicates=True)
        self.assertEqual(index[0][mask[0]].tolist(),[2,3])
        self.assertFalse(bool((index[0][mask[0]]==1).any()))

    def test_graph_immutable_when_evaluation_vocabulary_changes(self):
        m=model();state=model_state_hash(m);q=m.source_q.clone()
        m.set_classnames(['unseen','class1'],['a photo of unseen land.','a photo of land class1.'])
        self.assertEqual(tuple(m(torch.randn(2,3,8,8))['logits'].shape),(2,2))
        self.assertTrue(torch.equal(q,m.source_q));self.assertEqual(model_state_hash(m),state)
        with self.assertRaisesRegex(ValueError,'original source'):
            m.loss_terms(torch.randn(2,3,8,8),torch.tensor([0,1]))

    def test_graph_state_tampering_rejected(self):
        m=model();state=copy.deepcopy(m.method_state_dict())
        state['buffers']['source_delta']+=.1
        with self.assertRaises(ValueError):m.validate_method_state(state)

    def test_only_full_method_is_exposed(self):
        with self.assertRaises(ValueError):build_model('unsupported',None,None,{})

if __name__=='__main__':unittest.main()
