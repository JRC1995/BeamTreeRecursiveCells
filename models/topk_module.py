import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import Module
import numpy as np
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import svds


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()

    v = torch.ones([bs, 1, k_]) / (k_)
    G = torch.exp(-C / epsilon)

    v = v.to(C.device)

    for i in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    k = k_ - 1

    f = torch.zeros([bs, n, 1]).to(C.device)
    g = torch.zeros([bs, 1, k + 1]).to(C.device)

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)  # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa)  # [bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)  # [bs, n, k]
    G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma  # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  # [bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma  # [bs, n, k+1]
    G23 = - F.pad(g1_L, pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = - L.matmul(g2) * Gamma  # [bs, n, k+1]
    G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, n, k+1]
    #            del g2, G31, G32, L

    grad_C = (-G1 + G2 + G3) / epsilon  # [bs, n, k+1]

    return grad_C


class TopKFunc1(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom1(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(TopK_custom1, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors.to(scores.device)) ** 2
        C = C / (C.max().detach())
        # print(C)
        mu = torch.ones([1, n, 1], requires_grad=False).to(scores.device) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).to(scores.device) .view([1, 1, self.k + 1])

        Gamma = TopKFunc1.apply(C, mu, nu, self.epsilon, self.max_iter)
        # print(Gamma)
        A = Gamma[:, :, :self.k] * n

        return A, None


############################################################################

class TopKFunc2(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, Gamma0, weight1, weight2):
        bs, n, k_ = C.size()

        with torch.no_grad():

            b = (C + epsilon * torch.log(Gamma0)).view([bs, -1])
            x = weight2.matmul(b.unsqueeze(-1))
            x = weight1.matmul(x).squeeze(-1)

            f = x[:, :n].unsqueeze(-1)
            g = x[:, n:].unsqueeze(-2)
            g = torch.cat((g, torch.zeros([bs, 1, 1])), dim=-1)

            if torch.cuda.is_available():
                f = f.cuda()
                g = g.cuda()

            epsilon_log_mu = epsilon * torch.log(mu)
            epsilon_log_nu = epsilon * torch.log(nu)

            def min_epsilon_row(Z, epsilon):
                return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

            def min_epsilon_col(Z, epsilon):
                return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

            for i in range(max_iter):
                f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
                g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu
                # print(i, torch.cuda.memory_allocated()/1024.**3)

            Gamma = torch.exp((-C + f + g) / epsilon)

            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)

        return grad_C, None, None, None, None, None, None, None


class TopK_custom2_w_initial(torch.nn.Module):
    def __init__(self, k, n, epsilon=0.1, max_iter=200):
        super(TopK_custom2_w_initial, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

        self.tau = torch.ones(1) * 0.1
        self.tau.requires_grad = True

        rank = 5
        assert (rank <= n)

        A0 = sparse.hstack([sparse.diags(np.ones([n], dtype=np.float32)) for i in range(k + 1)]).reshape(n * (k + 1), n)
        A1 = sparse.vstack([sparse.diags(np.ones([k + 1], dtype=np.float32)) for i in range(n)])
        A = sparse.hstack([A0, A1]).tocsr()[:, :-1]
        u, s, vt = svds(A, rank)
        self.weight1 = (vt.dot(sparse.diags(1. / s))).todense()
        self.weight2 = u.T.todense()
        print('constructed the weight matrix')
        self.weight1 = torch.FloatTensor(self.weight1).unsqueeze(0)
        self.weight2 = torch.FloatTensor(self.weight2).unsqueeze(0)

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()
            self.tau = self.tau.cuda()
            self.weight1 = self.weight1.cuda()
            self.weight2 = self.weight2.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())
        # print(C)
        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        small_number = 1e-20
        topk_value, topk_indices = scores.detach().squeeze(-1).topk(self.k)
        #        Gamma0 = torch.zeros([bs, n, self.k]).scatter(-2, topk_indices.unsqueeze(-2), 1)
        Gamma0 = torch.sigmoid(-torch.abs(topk_value.unsqueeze(-2) - scores.repeat([1, 1, self.k])) / self.tau)
        Gamma0 = Gamma0 + small_number
        Gamma0 = Gamma0 / Gamma0.sum(-2, keepdim=True) / n
        Gamma0_last = (1. / n - Gamma0.sum(-1, keepdim=True)).clamp(small_number, 1 - small_number)
        Gamma0 = torch.cat((Gamma0, Gamma0_last), dim=-1)
        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()
            Gamma0 = Gamma0.cuda()

        Gamma = TopKFunc2.apply(C, mu, nu, self.epsilon, self.max_iter,
                                Gamma0, self.weight1, self.weight2)

        A = Gamma[:, :, :self.k] * n

        return A, torch.norm(Gamma - Gamma0)


############################################################################

class TopKFunc3(Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, Gamma0):
        ctx.save_for_backward(mu, nu, Gamma0)
        ctx.epsilon = epsilon

        return Gamma0

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)

        return grad_C, None, None, None, None, None, None, None


class TopK_custom3_only_initial(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(TopK_custom3_only_initial, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

        self.tau = torch.ones(1) * 0.1
        self.tau.requires_grad = True

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()
            self.tau = self.tau.cuda()

    def forward(self, scores):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())
        # print(C)
        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        small_number = 1e-10
        topk_value, topk_indices = scores.detach().squeeze(-1).topk(self.k)
        #        Gamma0 = torch.zeros([bs, n, self.k]).scatter(-2, topk_indices.unsqueeze(-2), 1)
        Gamma0 = torch.sigmoid(-torch.abs(topk_value.unsqueeze(-2) - scores.repeat([1, 1, self.k])) / self.tau)
        Gamma0 = Gamma0 + small_number
        Gamma0 = Gamma0 / Gamma0.sum(-2, keepdim=True) / n
        Gamma0_last = (1. / n - Gamma0.sum(-1, keepdim=True))
        Gamma0 = torch.cat((Gamma0, Gamma0_last), dim=-1).clamp(small_number, 1 - small_number)

        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()
            Gamma0 = Gamma0.cuda()

        Gamma = TopKFunc3.apply(C, mu, nu, self.epsilon, self.max_iter, Gamma0)

        A = Gamma[:, :, :self.k] * n

        return A, None


############################################################################

class TopKFunc4(Function):

    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, Gamma0, use_iter):

        with torch.no_grad():

            if use_iter:
                if epsilon > 1e-2:
                    Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                    if bool(torch.any(Gamma != Gamma)):
                        print('Nan appeared in Gamma, re-computing...')
                        Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
                else:
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)

            else:
                Gamma = Gamma0

            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon

        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)

        return grad_C, None, None, None, None, None, None


class TopK_custom4_train_tau(torch.nn.Module):

    def __init__(self, k, n, epsilon=0.1, max_iter=200):
        super(TopK_custom4_train_tau, self).__init__()

        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

        self.W = torch.randn(1, n, 1)

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()
            self.W = self.W.cuda()
        self.W.requires_grad = True

    def forward(self, scores, use_iter=True):
        bs, n = scores.size()
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())
        # print(C)
        mu = torch.ones([1, n, 1], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1])

        self.W.requires_grad = use_iter

        sorted_scores, _ = torch.sort(scores.detach(), dim=1)
        tau = (sorted_scores * self.W).sum(-2, keepdim=True)

        small_number = 1e-20
        topk_value, topk_indices = scores.detach().squeeze(-1).topk(self.k)
        #        Gamma0 = torch.zeros([bs, n, self.k]).scatter(-2, topk_indices.unsqueeze(-2), 1)
        Gamma0 = torch.sigmoid(-torch.abs(topk_value.unsqueeze(-2) - scores.repeat([1, 1, self.k])) / tau)
        Gamma0 = Gamma0 + small_number
        Gamma0 = Gamma0 / Gamma0.sum(-2, keepdim=True) / n
        Gamma0_last = (1. / n - Gamma0.sum(-1, keepdim=True)).clamp(small_number, 1 - small_number)
        Gamma0 = torch.cat((Gamma0, Gamma0_last), dim=-1)
        if torch.cuda.is_available():
            mu = mu.cuda()
            nu = nu.cuda()
            Gamma0 = Gamma0.cuda()

        Gamma = TopKFunc4.apply(C, mu, nu, self.epsilon, self.max_iter, Gamma0, use_iter)

        A = Gamma[:, :, :self.k] * n

        if use_iter:
            return A, torch.norm(Gamma - Gamma0)
        else:
            return A, None


############################################################################

class TopK_stablized(torch.nn.Module):
    def __init__(self, k, epsilon=0.1, max_iter=200):
        super(TopK_stablized, self).__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, k + 1, 1])
        self.max_iter = max_iter

        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

    def forward(self, scores):
        bs, n = scores.size()[:2]
        scores = scores.view([bs, 1, n])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors) ** 2
        C = C / (C.max().detach())
        f = torch.zeros([bs, 1, n])
        g = torch.zeros([bs, self.k + 1, 1])
        mu = torch.ones([1, 1, n], requires_grad=False) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, self.k + 1, 1])

        if torch.cuda.is_available():
            f = f.cuda()
            g = g.cuda()
            mu = mu.cuda()
            nu = nu.cuda()

        def min_epsilon_row(Z, epsilon):
            return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

        def min_epsilon_col(Z, epsilon):
            return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

        for i in range(self.max_iter):
            f = min_epsilon_col(C - f - g, self.epsilon) + f + self.epsilon * torch.log(mu)
            g = min_epsilon_row(C - f - g, self.epsilon) + g + self.epsilon * torch.log(nu)

        P = torch.exp((-C + f + g) / self.epsilon)
        A = P[:, :self.k, :] * n
        return A.transpose(-1, -2)


if __name__ == '__main__':
    torch.manual_seed(1)
    num_iter = int(1e3)
    k = 1
    epsilon = 1e-1
    soft_topk = TopK_custom1(k, epsilon=epsilon, max_iter=num_iter)
    soft_topk_test = TopK_stablized(k, epsilon=epsilon, max_iter=num_iter)
    #    scores = Parameter(torch.randn(1, 2))
    scores = Parameter(torch.FloatTensor([range(10)]))
    print('======scores======')
    print(scores)

    print('======topk_auto-diff======')
    A2 = soft_topk_test(scores)
    print(A2)

    loss2 = torch.sum(A2 ** 2)  # a dummy loss function
    loss2.backward()
    A2_grad = scores.grad.clone()
    print('======grad_w.r.t_score_auto-diff======')
    print(A2_grad)

    A, _ = soft_topk(scores)
    print('======topk_manual-grad======')
    print(A)

    scores.grad.data.zero_()
    loss = torch.sum(A ** 2)  # a dummy loss function
    loss.backward()
    A_grad = scores.grad.clone()
    print('======grad_w.r.t_score_manual-grad======')
    print(A_grad)

    print('======The diff between backward pass======')
    print(torch.norm(A_grad - A2_grad))
