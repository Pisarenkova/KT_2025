import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

a, b = 3, 4

Nx, Ny = 100, 100

x = np.linspace(-np.pi, np.pi, Nx)
y = np.linspace(-np.pi, np.pi, Ny)
X, Y = np.meshgrid(x, y)

Z = np.sin(a*X)*np.sin(b*Y)

#Шумы
gaussian_noise = 0.05 * np.random.normal(size=Z.shape)
u1 = Z + gaussian_noise

poisson_noise = np.random.poisson(0.05, Z.shape)
u2 = Z + poisson_noise

abs_diff_u1 = np.abs(Z - u1)
abs_diff_u2 = np.abs(Z - u2)

h_min, h_max, v_min, v_max = -np.pi, np.pi, -np.pi, np.pi

def create_subplot(ax, data, title, cmap):
    c = ax.imshow(data, extent=(h_min, h_max, v_min, v_max), origin='lower', cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    return c

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 0.04, 1, 0.04],hspace=0.3)

#Графики
create_subplot(fig.add_subplot(gs[0, 0]), Z, 'a)', cmap='RdBu_r')
create_subplot(fig.add_subplot(gs[1, 1]), u2, 'e)', cmap='RdBu_r')
c4 = create_subplot(fig.add_subplot(gs[0, 3]), abs_diff_u1, 'c)', cmap='inferno')
c5 = create_subplot(fig.add_subplot(gs[1, 3]), abs_diff_u2, 'f)', cmap='inferno')
create_subplot(fig.add_subplot(gs[1, 0]), Z, 'd)', cmap='RdBu_r')

#На втором графике отметки
c2_ax = fig.add_subplot(gs[0, 1])
c2_ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
c2_ax.set_xticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
c2_ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
c2_ax.set_yticklabels(['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
c2_ax.set_title('b)', fontsize=12)
c2_ax.set_xlabel('x', fontsize=12)
c2_ax.set_ylabel('y', fontsize=12)
c2 = c2_ax.imshow(u1, extent=(h_min, h_max, v_min, v_max), origin='lower', cmap='RdBu_r')

#Общий колорбар
l_b = min(min(np.min(Z),np.min(u1)),np.min(u2))  # Используем np.min вместо min
u_b = max(max(np.max(Z),np.max(u1)),np.max(u2))  # Используем np.max вместо max
norm_1= plt.Normalize(l_b, u_b)
cbar = fig.colorbar(c2, cax=fig.add_subplot(gs[:, 2]), norm=norm_1)
cbar.set_ticks(np.linspace(l_b, u_b, 5).round(2))

#колорбары справа
def add_colorbar(c, ax, data):
    norm = plt.Normalize(np.min(data), np.max(data))
    cbar = fig.colorbar(c, cax=ax, norm=norm)
    cbar.set_ticks(np.linspace(np.min(data), np.max(data), 7).round(3))
    cbar.set_label(r'$\times 10^{-3}$', rotation=0, labelpad=-2, loc='top')
    cbar.ax.yaxis.set_label_position('left')

add_colorbar(c4, fig.add_subplot(gs[0, 4]), abs_diff_u1)
add_colorbar(c5, fig.add_subplot(gs[1, 4]), abs_diff_u2)

#Сохранение итогов
plt.savefig('output_image.png', dpi=300, bbox_inches='tight')
#plt.tight_layout()
plt.show()