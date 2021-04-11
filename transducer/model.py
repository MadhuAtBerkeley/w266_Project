import torch

joiner_dim = 512
encoder_dim = 512
predictor_dim = 512



class Encoder(torch.nn.Module):
  def __init__(self, num_inputs):
    super(Encoder, self).__init__()
    
    #self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
    self.rnn = torch.nn.GRU(input_size=num_inputs, hidden_size=encoder_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)
    self.linear = torch.nn.Linear(encoder_dim*2, joiner_dim)

  def forward(self, x):
    out = self.rnn(x)[0]
    out = self.linear(out)
    return out



class Predictor(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Predictor, self).__init__()
    self.blank_index = num_outputs-1
    self.embed = torch.nn.Embedding(num_outputs, 32)
    self.rnn = torch.nn.GRUCell(input_size=32, hidden_size=predictor_dim)
    self.linear = torch.nn.Linear(predictor_dim, joiner_dim)
    
    self.initial_state = torch.nn.Parameter(torch.randn(predictor_dim))
    self.start_symbol = self.blank_index # In the original paper, a vector of 0s is used; just using the null index instead is easier when using an Embedding layer.

  def forward_one_step(self, input, previous_state):
    embedding = self.embed(input)
    state = self.rnn.forward(embedding, previous_state)
    out = self.linear(state)
    return out, state

  def forward(self, y):
    batch_size = y.shape[0]
    U = y.shape[1]
    outs = []
    state = torch.stack([self.initial_state] * batch_size).to(y.device)
    for u in range(U+1): # need U+1 to get null output for final timestep 
      if u == 0:
        decoder_input = torch.tensor([self.start_symbol] * batch_size).to(y.device)
      else:
        decoder_input = y[:,u-1]
      out, state = self.forward_one_step(decoder_input, state)
      outs.append(out)
    out = torch.stack(outs, dim=1)
    return out



class Joiner(torch.nn.Module):
  def __init__(self, num_outputs):
    super(Joiner, self).__init__()
    self.linear = torch.nn.Linear(joiner_dim, num_outputs)

  def forward(self, encoder_out, predictor_out):
    out = encoder_out + predictor_out
    out = torch.nn.functional.relu(out)
    out = self.linear(out)
    return out



class Transducer(torch.nn.Module):
  def __init__(self, num_inputs, num_outputs):
    super(Transducer, self).__init__()
    self.blank_index = num_outputs-1
    self.encoder = Encoder(num_inputs)
    self.predictor = Predictor(num_outputs)
    self.joiner = Joiner(num_outputs)
    

    if torch.cuda.is_available(): self.device = "cuda"
    else: self.device = "cpu"
    self.to(self.device)

  def compute_forward_prob(self, joiner_out, T, U, y):
    """
    joiner_out: tensor of shape (B, T_max, U_max+1, #labels)
    T: list of input lengths
    U: list of output lengths 
    y: label tensor (B, U_max+1)
    """
    B = joiner_out.shape[0]
    T_max = joiner_out.shape[1]
    U_max = joiner_out.shape[2] - 1
    log_alpha = torch.zeros(B, T_max, U_max+1).to(y.device)
    for t in range(T_max):
      for u in range(U_max+1):
          if u == 0:
            if t == 0:
              log_alpha[:, t, u] = 0.

            else: #t > 0
              log_alpha[:, t, u] = log_alpha[:, t-1, u] + joiner_out[:, t-1, 0, self.blank_index] 
                  
          else: #u > 0
            if t == 0:
              log_alpha[:, t, u] = log_alpha[:, t,u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
            
            else: #t > 0
              log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                  log_alpha[:, t-1, u] + joiner_out[:, t-1, u, self.blank_index],
                  log_alpha[:, t, u-1] + torch.gather(joiner_out[:, t, u-1], dim=1, index=y[:,u-1].view(-1,1) ).reshape(-1)
              ]), dim=0)
    
    log_probs = []
    for b in range(B):
      log_prob = log_alpha[b, T[b]-1, U[b]] + joiner_out[b, T[b]-1, U[b], self.blank_index]
      log_probs.append(log_prob)
    log_probs = torch.stack(log_probs) 
    return log_prob

  def compute_loss(self, x, y, T, U):
    encoder_out = self.encoder.forward(x)
    predictor_out = self.predictor.forward(y)
    joiner_out = self.joiner.forward(encoder_out.unsqueeze(2), predictor_out.unsqueeze(1)).log_softmax(3)
    loss = -self.compute_forward_prob(joiner_out, T, U, y).mean()
    return loss
    
  def greedy_search(self, x, T):
     y_batch = []
     B = len(x)
     print(B, len(T))  

     encoder_out = self.encoder.forward(x)
     U_max = 200
     for b in range(B):
        t = 0; u = 0; y = [self.predictor.start_symbol]; predictor_state = self.predictor.initial_state.unsqueeze(0)
        while t < T[b] and u < U_max:
           predictor_input = torch.tensor([ y[-1] ]).to(x.device)
           g_u, predictor_state = self.predictor.forward_one_step(predictor_input, predictor_state)
           f_t = encoder_out[b, t]
           h_t_u = self.joiner.forward(f_t, g_u)
           argmax = h_t_u.max(-1)[1].item()
           if argmax == self.blank_index:
              t += 1
           else: # argmax == a label
              u += 1
              y.append(argmax)
        y_batch.append(y[1:]) # remove start symbol
     return y_batch   





