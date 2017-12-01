from logger import Logger, struct
import matplotlib.pyplot as plt
# from pprint import pprint

H = struct()
S = struct()

log = Logger('mnist_momentum', H, S, load=True)
losses = [m.best_loss for m in log.metrics()]
# pprint(log.metrics())
# pprint(H.momenta)
plt.plot(H.momenta, losses)
plt.xscale('log')
plt.yscale('log')
plt.show()
