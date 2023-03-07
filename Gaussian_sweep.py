from toy_match_recursive import task
import numpy as np

### Use specific hyperparameters for a single run
lr_range = [1e-1, 1e-2, 1e-3, 1e-4]
p_range = [0.1, 1, 10]
on_range = [1, 10]
rep = 5
### Use specific hyperparameters for a single run

if __name__ == "__main__":

    test_acc = np.zeros((7, 5, 5))
    hyp = []
    for e in range(5):
        n = 3 * (e+1)

        # ERM
        # Select best hyperparameters
        max_acc = -1
        best_hyp = None
        for p in lr_range:
            args = dict(num_env=n, lr=p, layer=1, plambda=0, onlambda=0, coral=False, irm=False, all=False, seq=False)
            acc = task(args, 0)
            if acc > max_acc:
                max_acc = acc
                best_hyp = args.copy()
                break
        hyp.append(best_hyp)
        test_acc[0, e, 0] = max_acc
        # Run multiple random seeds, skipping runs that fail to converge
        r = 1
        sd = r
        while r < rep:
            test_acc[0, e, r] = task(best_hyp, sd)
            if test_acc[0, e, r] >= 0:
                r += 1
            sd += 1

        # IRM
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(num_env=n, lr=p2, layer=1, plambda=p1, onlambda=0, coral=False, irm=True, all=False, seq=False)
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
        sd = r
        while r < rep:
            test_acc[1, e, r] = task(best_hyp, sd)
            if test_acc[1, e, r] >= 0:
                r += 1
            sd += 1

        # CORAL
        max_acc = -1
        last_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p2 in lr_range:
                args = dict(num_env=n, lr=p2, layer=1, plambda=p1, onlambda=0, coral=True, irm=False, all=False, seq=False)
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
        r = 1
        sd = r
        while r < rep:
            test_acc[2, e, r] = task(best_hyp, sd)
            if test_acc[2, e, r] >= 0:
                r += 1
            sd += 1

        # CORAL + ON
        max_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p3 in on_range:
                for p2 in lr_range:
                    args = dict(num_env=n, lr=p2, layer=1, plambda=p1, onlambda=p3, coral=True, irm=False, all=False, seq=False)
                    acc = task(args, 0)
                    if acc > max_acc:
                        max_acc = acc
                        best_hyp = args.copy()
                    if acc > -1:
                        break
        hyp.append(best_hyp)
        test_acc[3, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[3, e, r] = task(best_hyp, r)

        # IFM 1-layer
        max_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p3 in on_range:
                for p2 in lr_range:
                    args = dict(num_env=n, lr=p2, layer=1, plambda=p1, onlambda=p3, coral=False, irm=False, all=False, seq=True)
                    acc = task(args, 0)
                    if acc > max_acc:
                        max_acc = acc
                        best_hyp = args.copy()
                        break
        hyp.append(best_hyp)
        test_acc[4, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[4, e, r] = task(best_hyp, r)

        # IFM 3-layer match-all
        max_acc = -1
        best_hyp = None
        for p1 in p_range:
            for p3 in on_range:
                for p2 in lr_range:
                    args = dict(num_env=n, lr=p2, layer=3, plambda=p1, onlambda=p3, coral=False, irm=False, all=True, seq=True)
                    acc = task(args, 0)
                    if acc > max_acc:
                        max_acc = acc
                        best_hyp = args.copy()
                        break
        hyp.append(best_hyp)
        test_acc[5, e, 0] = max_acc
        for r in range(1, rep):
            test_acc[5, e, r] = task(best_hyp, r)

        # IFM 3-layer match-disjoint
        if e > 0:
            max_acc = -1
            best_hyp = None
            for p1 in p_range:
                for p3 in on_range:
                    for p2 in lr_range:
                        args = dict(num_env=n, lr=p2, layer=3, plambda=p1, onlambda=p3, coral=False, irm=False, all=False, seq=True)
                        acc = task(args, 0)
                        if acc > max_acc:
                            max_acc = acc
                            best_hyp = args.copy()
                            break
            hyp.append(best_hyp)
            test_acc[6, e, 0] = max_acc
            for r in range(1, rep):
                test_acc[6, e, r] = task(best_hyp, r)

    print('Best hyperparameters', hyp)
    print(test_acc)
    print(np.mean(test_acc, axis=-1))
    print(np.std(test_acc, axis=-1))
