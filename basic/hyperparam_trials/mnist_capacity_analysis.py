from logger import Logger, struct
import matplotlib.pyplot as plt
from pprint import pprint

H = struct()
S = struct()
log = Logger('mnist_capacity', H, S, load=True)
losses = [m.best_loss for m in log.metrics()]

H_nodrop = struct()
S_nodrop = struct()
log_nodrop = Logger('mnist_capacity_nodrop', H_nodrop, S_nodrop, load=True)
losses_nodrop = [m.best_loss for m in log_nodrop.metrics()]

H_mlp = struct()
S_mlp = struct()
log_mlp = Logger('mnist_capacity_mlp', H_mlp, S_mlp, load=True)
losses_mlp = [m.best_loss for m in log_mlp.metrics()]

pprint(log.metrics())
pprint(log_nodrop.metrics())
# pprint(H.caps)
plt.plot(H.caps, losses, label='dropout')
plt.plot(H_nodrop.caps, losses_nodrop, label='nodrop')
plt.plot(H_mlp.caps, losses_mlp, label='mlp')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.show()
