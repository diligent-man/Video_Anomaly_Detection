__all__ = [
    "on_pretrain_routine_start",
    "on_pretrain_routine_end",
    "on_train_start",
    "on_train_epoch_start",
    "on_train_batch_start",
    "optimizer_step",
    "on_before_zero_grad",
    "on_train_batch_end",
    "on_train_epoch_end",
    "on_fit_epoch_end",
    "on_model_save",
    "on_train_end",
    "on_params_update",
    "teardown"
]


def on_pretrain_routine_start(trainer): pass
def on_pretrain_routine_end(trainer): pass
def on_train_start(trainer): pass
def on_train_epoch_start(trainer): pass
def on_train_batch_start(trainer): pass
def optimizer_step(trainer): pass
def on_before_zero_grad(trainer): pass
def on_train_batch_end(trainer): pass
def on_train_epoch_end(trainer): pass
def on_fit_epoch_end(trainer): pass
def on_model_save(trainer): pass
def on_train_end(trainer): pass
def on_params_update(trainer): pass
def teardown(trainer): pass
