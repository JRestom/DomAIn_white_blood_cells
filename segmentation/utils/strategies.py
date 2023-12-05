from avalanche.training import Naive, JointTraining, EWC, Replay, Cumulative, LwF


def strategy_loader(strategy, model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device='cuda'):
    # print(f"using {strategy} strategy ....")
    if strategy=='Naive':
        cl_strategy = Naive(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size, device)

    elif strategy=='Joint':
        cl_strategy = JointTraining(model,optimizer,criterion,train_mb_size,train_epochs,eval_mb_size,device)

    elif strategy=='EWC':
        cl_strategy = EWC(model,optimizer,criterion, ewc_lambda=1.0e-1, train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device)

    elif strategy=='Replay':
        cl_strategy = Replay(model, optimizer, criterion, mem_size=6, train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device)
    
    elif strategy=='Cumulative':
        cl_strategy = Cumulative(model, optimizer, criterion, train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device)
    
    elif strategy=='LwF':
        cl_strategy = LwF(model, optimizer, criterion, train_mb_size=train_mb_size, train_epochs=train_epochs, eval_mb_size=eval_mb_size, device=device)

    return cl_strategy



