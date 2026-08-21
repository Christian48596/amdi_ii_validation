import torch
from learned_amdi.networks import masked_categorical

def test_joint_log_probability_is_sum_of_local_log_probabilities():
    logits=torch.tensor([[0.2,0.3,0.5],[1.0,-1.0,0.0]],dtype=torch.float32)
    masks=torch.tensor([[1,1,1],[0,1,1]],dtype=torch.bool)
    actions=torch.tensor([2,1])
    d=masked_categorical(logits,masks)
    assert torch.allclose(d.log_prob(actions).sum(), d.log_prob(actions)[0]+d.log_prob(actions)[1])
