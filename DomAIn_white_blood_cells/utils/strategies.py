from avalanche.training import Naive, JointTraining, EWC, Replay


def strategy_loader(strategy, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device='cuda'):
    print(f"using {strategy} strategy ....")
    if strategy=='Naive':
        cl_strategy = Naive(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device)

    elif strategy=='Joint':
        cl_strategy = JointTraining(model,optimizer,criterion,train_mb_size,train_epochs,eval_mb_size,device)

    elif strategy=='EWC':
        cl_strategy = EWC(model,optimizer,criterion,train_mb_size, train_epochs, eval_mb_size, device, ewc_lambda=1.0e-1)

    elif strategy=='Replay':
        cl_strategy = Replay(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device, mem_size=50)

    return cl_strategy