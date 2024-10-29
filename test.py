def test(model, testloader):
    model.eval()
    model.cuda()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(acc)
    return acc
