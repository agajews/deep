from logger import Logger, struct
import matplotlib.pyplot as plt

H = struct()
S = struct()

log = Logger('mnist_lr', H, S, load=True)
losses = [m.best_loss for m in log.metrics()]
plt.plot(H.lrs, losses)
plt.xscale('log')
plt.yscale('log')
plt.show()
