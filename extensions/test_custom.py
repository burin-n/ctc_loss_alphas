import torch
import ctc_custom


def ctc_loss(input, target, input_lengths, target_lengths, blank, reduction='none', zero_infinity=False, get_alpha=False):
    assert reduction == 'none'
    if not get_alpha:
        # loss: (batch, )
        return torch.nn.functional.ctc_loss(input, target, input_lengths, target_lengths, blank, reduction, zero_infinity)
    else:
        # loss:(batch,), alpha :(batch, input_lengths.max(), 2*target_legnths.max()+1 )
        return ctc_custom.ctc_loss_alpha(input, target, input_lengths, target_lengths, blank, zero_infinity)

T = 50
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes
# Initialize random batch of input vectors, for *size = (T,N,C)

input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

loss1 = ctc_loss(input, target, input_lengths, target_lengths, 0, get_alpha=False)
loss2, alpha = ctc_loss(input, target, input_lengths, target_lengths, 0, get_alpha=True)

assert (loss1 == loss2).all()

loss1.mean().backward()
loss2.mean().backward()
