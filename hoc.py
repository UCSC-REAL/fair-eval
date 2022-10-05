
import numpy as np
import torch
import random
import time

smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)



def get_T_global_min(args, record, max_step = None, T0 = None, p0 = None, lr = 0.1, NumTest = None, all_point_cnt = 15000):

    if max_step is None:
        max_step = args.max_iter
    if NumTest is None:
        NumTest =  args.G

    KINDS = args.num_classes
    all_point_cnt = np.min((all_point_cnt,int(len(record)*0.9)))
    print(f'Sample {all_point_cnt} instances in each round')
    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    for idx in range(NumTest):
        print(idx, flush=True)
        sel_loc = np.random.permutation(record.shape[1])[:3]
        record_sel = record[:, sel_loc]
        # print(f'sel_loc is {sel_loc}')
        cnt_y_3 = count_y_known2nn(KINDS, record_sel, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    args.device = set_device()
    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, args.device, max_step, T0, p0, lr = lr)

    E_calc = E_calc.cpu().numpy()
    P_calc = P_calc.cpu().numpy()
    return E_calc, P_calc

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

def distCosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - cosine distance
    return dist

def count_real(KINDS, T, P, mode, _device = 'cpu'):
    # time1 = time.time()
    P = P.reshape((KINDS, 1))
    p_real = [[] for _ in range(3)]

    p_real[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    # p_real[2] = torch.zeros((KINDS, KINDS, KINDS)).to(_device)
    p_real[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        p_real[1] = torch.cat([p_real[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with p_estimate
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                p_real[2][r][(i+r+KINDS)%KINDS] = temp33[r]
        else:
            p_real[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]


    temp = []       # adjust the order of the output (N*N), keeping consistent with p_estimate
    for p1 in range(KINDS):
        temp = torch.cat((p_real[1][p1, KINDS-p1:], p_real[1][p1, :KINDS-p1]))
        p_real[1][p1] = temp
    return p_real


def func(KINDS, p_estimate, T_out, P_out, N,step, LOCAL, _device):
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)       # define the loss
    
    P = smp(P_out)
    # loss = loss + 0.1*torch.norm(P.view(-1) - torch.tensor([0.51441996, 0.34073234, 0.08246922, 0.06237848]))
    # loss = loss + 0.1 * torch.norm(P[3]-0.1) +  0.1 * torch.norm(P[2]-0.1)
    # P = P_out
    T = smt(T_out)

    mode = random.randint(0, KINDS-1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N, N*N*N
    p_temp = count_real(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    weight = [1.0,1.0,1.0]
    # weight = [2.0,1.0,1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j]) #/ np.sqrt(N**j)
    
    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P+eps))/10

    return loss


def calc_func(KINDS, p_estimate, LOCAL, _device, max_step = 501, T0=None, p0 = None, lr = 0.1):
    # init
    # _device =  torch.device("cpu")
    N = KINDS
    eps = 1e-8
    if T0 is None:
        T = 1 * torch.eye(N) - torch.ones((N,N))
        # T[-1] = torch.ones(N)
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device = None) / N + torch.rand((N,1), device = None)*0.1     # P：0-9 distribution
        # P[2:] -= 5.0
        # P =  torch.tensor([0.4,0.4,0.1,0.1])
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr = lr)

    # train
    loss_min = 100.0
    T_rec = torch.zeros_like(T)
    P_rec = torch.zeros_like(P)

    time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(KINDS, p_estimate, T, P, N,step, LOCAL, _device)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        # if step % 100 == 0:
        #     print('loss {}'.format(loss))
        #     print(f'step: {step}  time_cost: {time.time() - time1}')
        #     print(f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
        #     print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
        #     # print(f'P {np.round((P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
        #     time1 = time.time()

    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def count_y(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    max_val = np.max(dist)
    am = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist,axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt

def count_y_known2nn(KINDS, label_list, cluster_sum=None):

    if cluster_sum is not None:
        sample = np.random.choice(range(label_list.shape[0]), cluster_sum, replace=False)
        label_list = label_list[sample]

    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)

    for i in range(cluster_sum):
        cnt[0][label_list[i][0]] += 1
        cnt[1][label_list[i][0]][label_list[i][1]] += 1
        cnt[2][label_list[i][0]][label_list[i][1]][label_list[i][2]] += 1

    return cnt