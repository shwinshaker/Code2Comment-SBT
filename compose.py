from dataloader import load_input_label
from lstm import build_model, Evaluation, check_cuda, DIM
from utils import String_Encoder
import torch
import numpy as np
from torch.nn import functional as F

def generate_music(model, encoder, length=100, 
                                   init_notes='<',
                                   sampling='random',
                                   temperature=1,
                                   device=torch.device('cpu')):
    
    if sampling == 'argmax':
        temperature = 1
    elif sampling == 'random':
        assert(temperature)
    else:
        raise KeyError('sampling tech not avaible!')

    # evaluation mode
    model.eval()
    
    # init_hidden
    # todo: should we initialize hidden layer here?
    # model.hidden = model._init_hidden(device=device)

    # convert initial notes to tensor
    init_seq = []
    for w in init_notes:
        init_seq.append(encoder.get_one_hot(w))
        
    init_seq = torch.tensor(init_seq, dtype=torch.float).to(device)
    init_seq = init_seq.view(1,len(init_seq),-1)
    
    def _get_indices(output, sampling=sampling, temperature=temperature):
        # temperature based sampling
        # high temperature means low deterministic
        # pick indices on the output probability by softmax
        dim = output.shape[1]
        opt_soft = F.softmax(output/temperature, dim=1).cpu().detach().numpy()
        inds = []
        probs = []
        for opt in opt_soft:
            assert(opt.shape==(DIM,))
            if sampling == 'random':
                ind = np.random.choice(dim, 1, p=opt).squeeze()
            elif sampling == 'argmax':
                ind = np.argmax(opt).squeeze()
            else:
                raise KeyError
            inds.append(ind)
            probs.append(opt[ind])
        return inds, probs
    
    def _to_input(output):
        # convert a lstm output to input
        # to feed back to the net
        characters = []
        inputs = []
        inds, probs = _get_indices(output)
        for ind in inds:
            character = encoder.get_character(ind)
            inputs.append(encoder.get_one_hot(character))
            characters.append(character)
        inputs = torch.tensor(inputs, dtype=torch.float).to(device)
        inputs = inputs.view(1, len(characters), -1)
        assert(inputs.shape[-1] == DIM)
        return characters, inputs, probs
    
    notes = []
    confs = [1.]
    notes.extend(list(init_notes))
    confs *= len(init_notes)
    with torch.no_grad():
        outputs = model(init_seq)
        characters, inputs, probs = _to_input(outputs)
        # record the last output <- predicted
        notes.append(characters[-1])
        confs.append(probs[-1])
        # pick the last output as next input
        input = inputs[:, -1, :].view(1, 1, -1)
        for _ in range(length):
            # loop production
            # output -> input -> output -> ..
            output = model(input)
            character, input, prob = _to_input(output)
            notes.extend(character)
            confs.extend(prob)
    
    return ''.join(notes), ' '.join(['%.2f' % f for f in confs])


def main():

    # hyperparameters
    # init note -> try anything, but start with <start>
    init_notes = '<QUE>\n# initialize parameters\n<ANS>'
    #<QUE>' #\nWhat is the meaning of life?\n<ANS>'
    sampling = 'random' # 'argmax'
    temperature = 0.5 # avaible if sampling is random
    length = 1000 # length of generated music sheet

    print('---> check cuda')
    use_cuda, device, extras = check_cuda()
    print('cuda: %s' % 'yes' if use_cuda else 'no')
    if use_cuda:
        loc = 'cuda'
    else:
        loc = 'cpu'

    print('----> loading setup')
    init = torch.load('init0.pth.tar', map_location=loc)
    encoder = init['encoder']
    hidden_size = init['hidden_size']
    model, _, _ = build_model(input_dim=encoder.length, hidden_dim=hidden_size, device=device)

    print('---> loading best model')
    path = 'model_best0.pth.tar'
    checkpoint = torch.load(path, map_location=loc)
    model.load_state_dict(checkpoint['model'])

    print('---> Music sheet generated to music.txt')
    notes, confs = generate_music(model, encoder, length=length,
                                                  init_notes=init_notes,
                                                  sampling=sampling,
                                                  temperature=temperature,
                                                  device=device)
    notes_s = [s.replace('\n', '\\n') for s in list(notes)]
    notes_r = ' '.join([n+':'+c for n, c in zip(notes_s, confs.split())])
    print()
    print(notes,end='\n\n')
    with open("music.txt", "w") as f:
        f.write(notes)
        f.write('\n------------------\n')
        f.write(notes_r)


if __name__ == "__main__":
    main()







