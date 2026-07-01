import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
import matplotlib.pyplot as plt
import wandb # comment out if not using Weights and Biases

# CODE FOR TWO-NEURON MODEL
# potential nonlinearities: identity (defined below), jax.nn.relu, jax.nn.sigmoid

def identity(x):
    return x

def sero_prob_rnn(params, x0, z, tau_x, nln):
    """
    Arguments:
    - params
    - x0
    - z
    - inputs
    - tau_x   : decay constant
    """

    J = params["recurrence_matrix"]       # N x N
    C = params["readout_weights"]   # O x N
    rb = params["readout_bias"]             # O
    B_xz = params["nm_x_weight"]         # N x dim_nm
    N = J.shape[0]

    def inh(w):
        return -jnp.abs(w)
    
     # enforce mutual inhibition btwn mpoa/bnst units in J
    J = J.at[ : N//2, N//2 : ].set( inh( J[ : N//2, N//2 : ] ) )
    J = J.at[ N//2 : , : N//2 ].set( inh( J[ N//2 : , : N//2 ] ) )

    # make sure components of z can only excite/inhibit one subpopulation of x
    B_xz = B_xz.at[ : N//2, 1].set(0)
    B_xz = B_xz.at[N//2 : , 0].set(0)
    B_xz = jnp.abs(B_xz)

    def _step(x, z):
        # update x
        xp = x # hold onto previous value
        x = (1.0 - (1. / tau_x)) * xp # decay term
        x += (1. / (tau_x)) * nln(J @ xp + B_xz @ z) # update term

        # calculate y
        y = jax.nn.softmax(C @ x + rb)

        return x, (y, x)

    _, (ys, xs) = lax.scan(_step, x0, z)

    return ys, xs

batched_sero_prob_rnn = vmap(sero_prob_rnn, in_axes=(None, None, 0, None, None))

def batched_sero_prob_rnn_loss(params, x0, batch_z, tau_x, batch_targets, batch_mask, nln):
    ys, _ = batched_sero_prob_rnn(params, x0, batch_z, tau_x, nln)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# random parameter initialization
def random_sero_prob_rnn_params(key, z, n, o):
    """Generate random parameters

    Arguments:
    u:  number of inputs
    z:  dimension of NM inputs
    n:  number of neurons in main network
    o:  number of outputs
    """
    skeys = jr.split(key, 5)
    pfactor = 1.0 / jnp.sqrt(n) # scaling 

    return {'recurrence_matrix' : jr.normal(skeys[0],(n,n))*0.1,
            'readout_weights' : jr.normal(skeys[1], (o,n))*pfactor,
            'readout_bias' : jr.normal(skeys[2], (o,))*pfactor,
            'nm_x_weight' : jr.normal(skeys[3], (n,z))*pfactor}

# script to fit model to data
def fit_sero_prob_rnn(zs, targets, loss_masks, params, optimizer, x0, num_iters, tau_x, nln=jax.nn.relu,
                   wandb_log=False): # training on full set of data
    opt_state = optimizer.init(params)

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(batched_sero_prob_rnn_loss)(params, x0, zs, tau_x, targets, loss_masks, nln)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []

    best_loss = 1e6
    best_params = params
    for n in range(num_iters//100):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=100) 
        losses.append(loss_values)
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses


# data generation, phasic serotonin
def data_generation(key, num_samples):
    keys = jr.split(key, 7)

    # random time that pups are placed into arena, between 15 and 25 timesteps
    pup_start = jr.randint(keys[0], (num_samples,), 15, 25)
    # u1 is 0 until pup placed in arena, then 1 after
    u1 = (jnp.arange(100) >= pup_start[:, None]).astype(jnp.float32)[:,:]
    
    # Create a grid of T indices
    t_indices = jnp.arange(100)

    # select which trials are virgins and which are mothers, 0 is virgin, 1 is mother
    mpoa_select = jr.randint(keys[1], (num_samples,), 0, 2).astype(bool)

        # Create a grid of T indices
    t_indices = jnp.arange(100)
    # Create a boolean mask of shape (N, T)
    # True where the index is between pup_start and pup_start + 20
    mask = (t_indices >= pup_start[:, None]) & (t_indices < pup_start[:, None] + 20)
    # Calculate the local index within the ramp for each 'True' position
    ramp_indices = t_indices - (pup_start[:, None])
    # Map ramp values (flipped for transient signal) into the shape (N, T) and apply the mask
    ramp = jnp.arange(0.05,1.05,step=1/20)
    placed_ramp = jnp.where(mask, jnp.flip(ramp)[ramp_indices], jnp.zeros_like(u1)) # ramp ends at 0

    z = jnp.where(mpoa_select[:,None,None], 
                  jnp.stack((placed_ramp+ jr.normal(keys[2], (num_samples, 100)) * 0.05, placed_ramp+ jr.normal(keys[3], (num_samples, 100)) * 0.05), axis=-1),
                  jnp.stack((placed_ramp+ jr.normal(keys[2], (num_samples, 100)) * 0.05, jr.normal(keys[3], (num_samples, 100)) * 0.05), axis=-1))

    p = jnp.zeros((num_samples, 100, 3))
    p = jnp.where(u1[:,:,None] == 0, jnp.array([0.005, 0.005, 0.99]), p)
    p = jnp.where((u1[:,:,None] == 1) & (mpoa_select[:,None,None] == 1), jnp.array([0.5, 0.25, 0.25]), p)
    p = jnp.where((u1[:,:,None] == 1) & (mpoa_select[:,None,None] == 0), jnp.array([0.25, 0, 0.75]), p)

    return z, p

# Weights and Biases model save function (uncomment if needed and add requisite packages)
# def log_wandb_model(model, name, type):
#     trained_model_artifact = wandb.Artifact(name,type=type)
#     if not os.path.isdir('models'): os.mkdir('models')
#     subdirectory = wandb.run.name
#     filepath = os.path.join('models', subdirectory)
#     try: os.mkdir(filepath)
#     except: filepath=filepath
#     obs_outfile = open(os.path.join(filepath, "model"), 'wb')
#     pickle.dump(model, obs_outfile)
#     obs_outfile.close()
#     trained_model_artifact.add_dir(filepath)
#     wandb.log_artifact(trained_model_artifact)

# parameters (can track in Weights and Biases by uncommenting below)
config = dict(
    # model parameters
    N = 2,    # hidden state dim
    nln = 'relu',
    # Model Hyperparameters
    tau = 10,
    # Training
    num_full_train_iters = 20_000,
    keyind = 13,
)

# projectname = ...
# wandb.init(config=default_config, project=projectname, entity=...)
# config = wandb.config

func_map = {
    "identity": identity,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
}

n_neurons = config['N']
keyn = config['keyind']
init_params = random_sero_prob_rnn_params(jr.PRNGKey(keyn), 2, n_neurons, 3)
z,p = data_generation(jr.PRNGKey(13), 100)

tau_x = config['tau']

optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-3),
)

x0 = jnp.ones((n_neurons,))*0.1
nln_func = func_map[config['nln']]

params, losses = fit_sero_prob_rnn(z, p, jnp.ones((100,100,1)), init_params, optimizer, x0, config["num_full_train_iters"], tau_x, nln=nln_func, wandb_log=False)

# Code below is for saving model to Weights and Biases and generating diagnostic figures

# log_wandb_model(params, "sero_rnn_n{}".format(config['N']), 'model')

# # Plot of J matrix
# J = params["recurrence_matrix"]
# N = J.shape[0]
# def inh(w):
#     return -jnp.abs(w)
# J = J.at[ : N//2, N//2 : ].set( inh( J[ : N//2, N//2 : ] ) )
# J = J.at[ N//2 : , : N//2 ].set( inh( J[ N//2 : , : N//2 ] ) )

# fig, ax = plt.subplots()
# im = ax.imshow(J, cmap="bwr", vmin=-5, vmax=5)
# ax.set_title("Learned J Matrix")
# fig.colorbar(im, ax=ax)

# wandb.log({'j_matrix': wandb.Image(fig)}, commit=True)

# for ind in [2,5]:
#     dict = {2: "mother", 5: "virgin"}

#     y, x = sero_prob_rnn(params, x0, z[ind], u[ind], tau_x, nln=nln_func)
    
#     # targets/output plot
#     fig, ax = plt.subplots()
#     ax.plot(y, label="RNN output")
#     ax.plot(p[ind], label="Target")
#     ax.plot(u[:,:,1][:,:,None][ind], label="inputs")
#     ax.plot(z[ind], label="nm signal")
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     ax.set_xlabel("Time")
#     plt.tight_layout()

#     wandb.log({f'model_output_{dict[ind]}': wandb.Image(fig)}, commit=True)

#     # activity plot
#     fig, ax = plt.subplots()
#     ax.plot(x[:,:N//2], c="blue", label="BNST")
#     ax.plot(x[:,N//2:], c="red", label="mPOA")
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.tight_layout()

#     wandb.log({f'model_activity_{dict[ind]}': wandb.Image(fig)}, commit=True)