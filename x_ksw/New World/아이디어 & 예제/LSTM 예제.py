def lossFun(inputs, targets, hprev, cprev):
    xs, hs, cs, is_, fs, os, gs, ys, ps= {}, {}, {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev) # t=0일때 t-1 시점의 hidden state가 필요하므로
    cs[-1] = np.copy(cprev)
    loss = 0
    H = hidden_size
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        tmp = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh  # hidden state
        is_[t] = sigmoid(tmp[:H])
        fs[t] = sigmoid(tmp[H:2 * H])
        os[t] = sigmoid(tmp[2 * H: 3 * H])
        gs[t] = np.tanh(tmp[3 * H:])
        cs[t] = fs[t] * cs[t-1] + is_[t] * gs[t]
        hs[t] = os[t] * np.tanh(cs[t])

    # compute loss
    for i in range(len(targets)):
        idx = len(inputs) - len(targets) + i
        ys[idx] = np.dot(Why, hs[idx]) + by  # unnormalized log probabilities for next chars
        ps[idx] = np.exp(ys[idx]) / np.sum(np.exp(ys[idx]))  # probabilities for next chars
        loss += -np.log(ps[idx][targets[i], 0])  # softmax (cross-entropy loss)

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])
    n = 1
    a = len(targets) - 1
    for t in reversed(range(len(inputs))):
        if n > len(targets):
            continue
        dy = np.copy(ps[t])
        dy[targets[a]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dc = dcnext + (1 - np.tanh(cs[t]) * np.tanh(cs[t])) * dh * os[t]  # backprop through tanh nonlinearity
        dcnext = dc * fs[t]
        di = dc * gs[t]
        df = dc * cs[t-1]
        do = dh * np.tanh(cs[t])
        dg = dc * is_[t]
        ddi = (1 - is_[t]) * is_[t] * di
        ddf = (1 - fs[t]) * fs[t] * df
        ddo = (1 - os[t]) * os[t] * do
        ddg = (1 - np.tanh(gs[t]) * np.tanh(gs[t])) * dg
        da = np.hstack((ddi.ravel(),ddf.ravel(),ddo.ravel(),ddg.ravel()))
        dWxh += np.dot(da[:,np.newaxis],xs[t].T)
        dWhh += np.dot(da[:,np.newaxis],hs[t-1].T)
        dbh += da[:, np.newaxis]
        dhnext = np.dot(Whh.T, da[:, np.newaxis])
        n += 1
        a -= 1
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]