import torch
import torch.nn as nn
from dataloader import createLoaders
import numpy as np
import time
import shutil

DIM = 96

class Evaluation():
    def __init__(self):
        self.epoch = 1
        self.loss = .0
        self.count_data = 0
        self.count_save = 0
        self.count_chunk = 0
        self.history = {}
        
    def reset(self, epoch):
        self.epoch = epoch
        self.loss = .0
        self.count_data = 0
        self.count_save = 0
        self.count_chunk = 0
        self.history[epoch] = []
        
    def __call__(self, loss, outputs):
        
        loss_ = loss.cpu().detach().numpy()
        outputs_ = outputs.cpu().detach().numpy().squeeze()
        chunk_size = outputs_.shape[0]
        self.loss += loss_ * chunk_size
        self.count_data += chunk_size
        self.count_chunk += 1
        
    def avg_loss(self):
        return self.loss / self.count_data
        
    def save(self, train_loss, val_loss):
        self.count_save += 1
        self.history[self.epoch].append((train_loss, val_loss))


# lstm model
class Composer(nn.Module):
    
    def __init__(self, dim=DIM, hidden_dim=100, device=None):
        super(Composer, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden_dim,
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, dim)
        self.hidden = self._init_hidden(device)
        
    def _init_hidden(self, device):
        return [torch.zeros([1, 1, self.hidden_dim]).to(device),
                torch.zeros([1, 1, self.hidden_dim]).to(device)]
    
    def forward(self, chunk):
        assert(chunk.shape[0]==1)
        # assert(chunk.shape[1]==100)
        assert(chunk.shape[2]==DIM)
        self.hidden = [h.detach() for h in self.hidden]
        output, self.hidden = self.lstm(chunk, self.hidden)
        opt_chunk = self.linear(output.view(chunk.shape[1], -1))
        return opt_chunk # output
           
 
def preprocessing(chunk_size=100):

    # load data
    loaders, encoder = createLoaders(extras=extras, chunk_size=chunk_size)
    dataloaders = dict(zip(['train', 'val', 'test'], loaders))
    print('------- Info ---------')
    for phase in dataloaders:
        print('- %s size: %i' % (phase, len(dataloaders[phase])))
    print('----------------------')
    
    return dataloaders, encoder


def build_model(input_dim=DIM, hidden_dim=100, learning_rate=0.1, device=None):
    
    model = Composer(dim=input_dim, hidden_dim=hidden_dim, device=device)
    # run on the gpu or cpu
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer
    
    
def train_model(model, criterion, optimizer, dataloaders, 
                num_epochs=1, best_loss=10, chunk_size=100,
                evaluate=Evaluation(), device=None, istest=False):
    # init timer
    since = time.time()
    start_epoch = evaluate.epoch
    step = 500 * 100 // chunk_size
    if istest: step = 10
    
    for epoch in range(start_epoch, num_epochs+1):
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        ## reset evaluator in a new epoch
        evaluate.reset(epoch)
        
        for i, (inputs, targets) in enumerate(dataloaders['train']):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()
            
            # regular stuff
            outputs = model(inputs)
            # squeeze the unnecessary batchsize dim
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
            
            # evaluation
            evaluate(loss, outputs)
            
            # validate every n chunks
            if i % step == 0:
                train_loss = evaluate.avg_loss()
                # validate first
                val_loss = validate_model(model, criterion,
                                          dataloaders['val'],
                                          istest=istest,
                                          device=device)
                
                # update best loss
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)
                
                # verbose
                print('[%i] '
                      'train-loss: %.4f '
                      'val-loss: %.4f '
                      '' % (evaluate.count_save,
                            train_loss,
                            val_loss))
                
                # save for plot
                evaluate.save(train_loss, val_loss)
                save_checkpoint({'model': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'best_loss': best_loss,
                                 'history': evaluate}, is_best)

            if istest:
                if i == 100: break
                      
                
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                

# could also be use to test
def validate_model(model, criterion, loader, device=None, verbose=False, istest=False):
    
    model.eval() # Set model to evaluate mode
    
    evaluate = Evaluation()
    step = 50
    if istest: step = 1

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(loader):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            evaluate(loss, outputs)

            if verbose:
                if j % step == 0:
                    print('[%i] val-loss: %.4f' % (j, evaluate.avg_loss()))

            if istest:
                if j == 2: break
            
    model.train() # Set model to training mode
    return evaluate.avg_loss()


def save_checkpoint(state, is_best):
    filename='checkpoint'+str(model_num)+'.pth.tar'
    bestname='model_best'+str(model_num)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def check_cuda():
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
    else: # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = False
    return use_cuda, device, extras


def main(learning_rate=0.01, hidden_size=100, chunk_size=100, device=None):

    # hyperparameters
    num_epochs = 50
    # learning_rate = 0.1
    # hidden_size = 100
    # chunk_size = 100

    print('------- Hypers --------\n'
          '- epochs: %i\n'
          '- learning rate: %g\n'
          '- hidden size: %i\n'
          '- chunk size: %i\n'
          '----------------'
          '' % (num_epochs, learning_rate, hidden_size, chunk_size))

    dataloaders, encoder = preprocessing(chunk_size=chunk_size)
    # save loader and encoder for later use
    torch.save({'loaders': dataloaders,
                'encoder': encoder,
                'hidden_size': hidden_size},
                'init'+str(model_num)+'.pth.tar')

    model, criterion, optimizer = build_model(input_dim=encoder.length, 
                                              hidden_dim=hidden_size,
                                              learning_rate=learning_rate,
                                              device=device)

    if resume:
        print('---> loading checkpoint')
        path = 'checkpoint'+str(model_num)+'.pth.tar'
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        evaluate = checkpoint['history']
        best_loss = checkpoint['best_loss']
    else:
        best_loss = 10 # anything as long as sufficiently large
        evaluate = Evaluation()

    train_model(model, criterion, optimizer, dataloaders,
                num_epochs=num_epochs, evaluate=evaluate, chunk_size=chunk_size,
                best_loss=best_loss, istest=debug, device=device)


if __name__ == "__main__":

    # global parameters
    torch.manual_seed(7)
    model_num = 0
    debug = True # debug mode
    resume = False # requires former checkpoint file

    use_cuda, device, extras = check_cuda()

    print('\n------- Globals --------\n'
          '- resume training: %s\n'
          '- debug mode: %s\n'
          '- # model: %i\n'
          '- cuda supported: %s\n'
          '------------------------'
          '' % ('yes' if resume else 'no', 
                'on' if debug else 'off',
                 model_num,
                'yes' if use_cuda else 'no'))

    main(device=device)
