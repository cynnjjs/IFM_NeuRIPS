from mnist_match_recursive import task
import numpy as np

### Use specific hyperparameters for a single run
lr_range = [1e-1, 1e-2, 1e-3]
p_range = [0.1, 1, 10, 100]
rep = 5
### Use specific hyperparameters for a single run

if __name__ == "__main__":
    test_acc = np.zeros((5, 6, 5))
    hyp = []

    for e in range(6):
        if e == 4:
            l = 3
            no_shr = True
        else:
            l = e+1
            no_shr = False

        # ERM
        max_acc = -1
        best_hyp = None
        for p in lr_range:
            args = dict(lr=p, layer=l, plambda=0, coral=False, irm=False, all=False, last=False, no_shrink=no_shr)
            acc = task(args, 0)
            if acc > max_acc:
                max_acc = acc
                best_hyp = args.copy()
                break
        hyp.append(best_hyp)
        test_acc[0, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[0, e, r] = task(best_hyp, r)
        """
        # IRM
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(lr=p2, layer=l, plambda=p1, coral=False, irm=True, all=True, last=True, no_shrink=no_shr)
                acc = task(args, 0)
                if acc > max_acc:
                    max_acc = acc
                    best_hyp = args.copy()
                if acc > -1:
                    break
            if acc < last_acc:
                break
            last_acc = acc
        hyp.append(best_hyp)
        test_acc[1, e, 0] = max_acc
        r = 1
        sd = 5
        while r < rep:
            test_acc[1, e, r] = task(best_hyp, sd)
            if test_acc[1, e, r] >= 0:
                r += 1
            sd += 1

        # CORAL - only match last layer
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(lr=p2, layer=l, plambda=p1, coral=True, irm=False, all=True, last=True, no_shrink=no_shr)
                acc = task(args, 0)
                if acc > max_acc:
                    max_acc = acc
                    best_hyp = args.copy()
                if acc > -1:
                    break
            if acc < last_acc:
                break
            last_acc = acc
        hyp.append(best_hyp)
        test_acc[2, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[2, e, r] = task(best_hyp, r)

        # CORAL match-disjoint
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(lr=p2, layer=l, plambda=p1, coral=True, irm=False, all=False, last=False, no_shrink=no_shr)
                acc = task(args, 0)
                if acc > max_acc:
                    max_acc = acc
                    best_hyp = args.copy()
                if acc > -1:
                    break
            if acc < last_acc:
                break
            last_acc = acc
        hyp.append(best_hyp)
        test_acc[3, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[3, e, r] = task(best_hyp, r)

        # CORAL match-all
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(lr=p2, layer=l, plambda=p1, coral=True, irm=False, all=True, last=False, no_shrink=no_shr)
                acc = task(args, 0)
                if acc > max_acc:
                    max_acc = acc
                    best_hyp = args.copy()
                if acc > -1:
                    break
            if acc < last_acc:
                break
            last_acc = acc
        hyp.append(best_hyp)
        test_acc[4, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[4, e, r] = task(best_hyp, r)
        """

    print('Best hyperparameters', hyp)
    print(test_acc)
    print(np.mean(test_acc, axis=-1))
    print(np.std(test_acc, axis=-1))
