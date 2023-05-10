from copy import deepcopy
# import torch
# import torch.nn as nn
# import torch.jit
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
import numpy
import paddle
import paddle.nn as nn
import paddle.jit
import paddle.nn.functional as F
# import paddle.fluid.layers as F
# import paddle.fluid.dygraph as nn
from functools import partial
import types

class Mop(nn.Layer):
    def __init__(self, model, optimizer,cfg):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = cfg.MODEL.EPISODIC
        self.var_scal = cfg.var_scal
        self.kl_par = cfg.kl_par
        self.entr_par = cfg.entr_par

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    
    def forward_and_adapt(self,x, model, optimizer):
        with paddle.set_grad_enabled(True):
            
            outputs = model(x)
            loss = self.entr_par * softmax_entropy(outputs).mean(0)
            kl_loss = self.kl_par * get_kl_loss(model,self.var_scal)
            # print('entr_par', self.entr_par)
            # print('kl_par', self.kl_par)
            # print('outputs', outputs)
            # print('kl_loss', kl_loss.item())
            # print('loss', loss.item())
            print('loss', loss.item(), 'kl_loss',kl_loss.item())
            loss += kl_loss
            
            loss.backward()
            # optimizer.step()
            optimizer.minimize(loss)
            optimizer.clear_grad()
            return outputs

# @paddle.jit.script
# @paddle.jit.to_static
def softmax_entropy(x: paddle.to_tensor) -> paddle.to_tensor:
    """Entropy of softmax distribution from logits."""
    m=nn.Softmax(axis=1)
    n=nn.LogSoftmax(axis=1)
    return -(m(x) * n(x)).sum(1)
    # return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def get_kl_loss(model,variance_scaling):
    # print(model.sublayers())
    modules = [x for x in model.sublayers() if hasattr(x, 'logvar0')]
    component1 = 0.0
    component2 = 0.0
    k = 0.0
    for x in modules:
        x.weight.stop_gradient = True
        k += x.weight.numel()
        p_var = variance_scaling * paddle.ones_like(x.weight, dtype='float32') * x.sigma0
        # print('pvar', p_var)
        # print('x.weight', x.weight.stop_gradient)
        # assert(False)
        # print('x.sigma0', x.sigma0)
        component1 += (p_var.log() - x.logvar0).sum()
        component2 += (x.logvar0.exp()/p_var).sum()
        # print('com2',(x.logvar0.exp()/p_var).shape)
    # print('x.sigma0', x.sigma0)
    # print('logv', x.logvar0, x.logvar0.stop_gradient)
    # print('xweight', x.weight, x.weight.stop_gradient)
    # print('component1', component1.item())
    # print('component2', component2.item())
    # print('k', k.item(), 'xsigma',x.sigma0, 'xweightshape',x.weight.shape)
    
    kl_loss = 0.5 *(component1 -k + component2)/k
    # print(kl_loss.stop_gradient, component1.stop_gradient, component2.stop_gradient)
    # print('kl_loss ', kl_loss)
    return kl_loss

def variational_forward(module, input):
    var = module.logvar0.expand_as(module.weight).exp()
    if isinstance(module, nn.Conv2D):
        output = F.conv2d(input,module.weight, module.bias, module._stride,
                          module._padding, module._dilation, module._groups)
        output_var = F.conv2d(input ** 2 + 1e-2, var, None, module._stride,
                              module._padding, module._dilation, module._groups)
    elif isinstance(module, nn.Linear):
        output = F.linear(input,  module.weight, module.bias)
        output_var = F.linear(input ** 2 + 1e-2, var, None)
    else:
        raise NotImplementedError("Module {} not implemented.".format(type(module)))
    # eps = F.normalize(paddle.empty_like(output))
    rand=paddle.distribution.Normal(loc=0,scale=1)
    eps = rand.sample(output.shape)
    return output + paddle.sqrt(output_var) * eps

def get_variational_vars(model):
    """Returns all variables involved in optimizing the hessian estimation."""
    result = []
    if hasattr(model, 'logvar0'):
        result.append(model.logvar0)
    for l in model.children():
        result += get_variational_vars(l)
    return result

def _add_logvar(module, args,is_bt=False):
    learn_dims = args.learn_dims
    variance_scaling = args.var_scal_init
    if not hasattr(module, 'weight'):
        return
    w = module.weight
    # print('w', w)
    if w.dim() < 2:
        return
    if not hasattr(module, 'logvar0'):
        if learn_dims == 'all' or is_bt:
            # print('start weight ', module.weight.shape)
            Var=paddle.reshape(w, [w.shape[0], -1])
            
            if isinstance(module, nn.Linear):
                Var=paddle.var(Var, axis=0)
                # print('after var', Var.shape)
                Var=paddle.reshape(Var, [*([1] * (w.dim() - 1)), -1])
                # print('view var', Var.shape)

            else:
                Var=paddle.var(Var, axis=1)
                Var=paddle.reshape(Var, [-1, *([1] * (w.dim() - 1))])
            logvar_expand = (Var * variance_scaling + 1e-10).log().clone().expand_as(module.weight)
            empty_var = paddle.ones_like(w, dtype='float32')
            create_parameter=module.create_parameter(empty_var.shape)
            module.add_parameter("logvar0", create_parameter)
            # module.logvar0 = nn.ParameterList(empty_var)
            # module.logvar0[:].set_value = logvar_expand
            module.logvar0.set_value(logvar_expand)
            # print('logvar stopgra',module.logvar0.stop_gradient)
            # print(1111111111111111111111111111111111111)
            # print('module.logvar0', module.logvar0)

        elif learn_dims == 'one':
            Var=paddle.reshape(w, [w.shape[0], -1])
            if isinstance(module, nn.Linear):
                Var=paddle.var(Var, axis=0)
                Var=paddle.reshape(Var, [*([1] * (w.dim() - 1)), -1])
            else:
                Var=paddle.var(Var, axis=1)
                Var=paddle.reshape(Var, [-1,*([1] * (w.dim() - 1))])
            # Var = w.reshape([w.size(0), -1]).var(dim=1).reshape([-1, *([1] * (w.dim() - 1))])

            create_parameter=module.create_parameter(Var.log().shape)
            module.add_parameter("logvar0", create_parameter)

            # for name in module.named_parameters():
            #     print(name)
            # module.logvar0[:].set_value = (Var * variance_scaling + 1e-10).log()
            module.logvar0.set_value((Var * variance_scaling + 1e-10).log())
            # print('logvar stopgra',module.logvar0.stop_gradient)
            print(22222222222222222222222222222)
        
        if isinstance(module, nn.Linear):
            module.sigma0 = paddle.var(w, axis=list(range(0,w.dim()-args.pr_dim_start)), keepdim=True).expand_as(w)
            # print('module.sigma0', module.sigma0)
            # print('w', w)
        else:
            module.sigma0 = paddle.var(w, axis=list(range(args.pr_dim_start,w.dim())), keepdim=True).expand_as(w)
        # module.sigma0 = w.var(dim=list(range(args.pr_dim_start,w.dim())),keepdim=True).expand_as(w)
        # module.sigma0.requires_grad = False
        module.sigma0.stop_gradient=True

def make_variational(model,args,is_bt=False):
    """Replaces the forward pass of the model layers to add noise."""
    model.apply(partial(_add_logvar, args=args,is_bt=is_bt))
    # i=0
    for m in model.sublayers():
        # print('makev',i, type(m))
        # i+=1
        if hasattr(m, 'logvar0'):
            m.forward = types.MethodType(variational_forward, m)

def variational_fisher(model, args):
    model.train()
    # model.requires_grad_(False)
    model.stop_gradient=True
    print("hook logvar...")
    parameters = []
    cls_parameters = []

    # i=0
    for m in model.layers[0:-1]:
        if isinstance(m, nn.Layer) and (not isinstance(m, nn.BatchNorm2D)):
            # print('vafisher', i, type(m))
            # i+=1
            make_variational(m,args)
            parameters += get_variational_vars(m)

    make_variational(model.layers[-1],args,is_bt=True)
    cls_parameters += get_variational_vars(model.layers[-1])

    for m in model.sublayers():
        if isinstance(m, nn.BatchNorm2D):
            # m.requires_grad_(True)
            m.stop_gradient=False
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            parameters += list(m.parameters())

    print("hook complete!")
    return parameters,cls_parameters
