def adjust_lr(optimizer, epoch, gamma):
    schedule = list(range(3,32,2))
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']
