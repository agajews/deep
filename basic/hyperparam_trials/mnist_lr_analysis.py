from logger import Logger, struct
import matplotlib.pyplot as plt
from pprint import pprint

H = struct()
S = struct()

log = Logger('mnist_lr2', H, S, load=True)
losses = [m.tn_loss for m in log.metrics()]
pprint(log.metrics())
pprint(H.lrs)
plt.plot(H.lrs, losses)
plt.xscale('log')
plt.yscale('log')
plt.show()
