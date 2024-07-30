# KelvinHelmholtz2D datapipe

KelvinHelmholtz2D datapipe是一個模擬工具，用於生成Kelvin-Helmholtz不穩定性問題的數據樣本。這些樣本是隨機初始條件下生成的，並用於測試和訓練數據驅動模型。

## 核心功能

1. **生成隨機初始條件**：
   - 使用隨機數生成初始擾動頻率，這些頻率被用於構建初始條件，模擬Kelvin-Helmholtz不穩定性。

2. **求解Euler方程**：
   - 使用有限體積法在GPU上求解Euler方程，通過這些方程模擬流體的不穩定性和演化過程。

3. **生成時間序列數據**：
   - 通過數值求解得到一系列快照，這些快照反映了流體隨時間演化的過程。

## 具體流程

### 1. 隨機生成初始擾動
- **隨機頻率生成**：使用隨機數生成初始擾動頻率，這些頻率用於構建初始條件。
- **初始化場**：將初始條件應用於密度、速度和壓力場。

```python
def initialize_batch(self) -> None:
    """Initializes arrays for new batch of simulations"""

    # initialize random Fourier freq
    seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
    wp.launch(
        init_uniform_random_2d,
        dim=[self.batch_size, self.nr_perturbation_freq],
        inputs=[self.w, -self.perturbation_range, self.perturbation_range, seed],
        device=self.device,
    )

    # initialize fields
    wp.launch(
        initialize_kelvin_helmoltz_batched_2d,
        dim=self.dim,
        inputs=[
            self.rho,
            self.vel,
            self.p,
            self.w,
            0.05 / np.sqrt(2.0),
            self.dim[1],
            self.dim[2],
            self.nr_perturbation_freq,
        ],
        device=self.device,
    )
    wp.launch(
        euler_primitive_to_conserved_batched_2d,
        dim=self.dim,
        inputs=[
            self.rho,
            self.vel,
            self.p,
            self.mass,
            self.mom,
            self.e,
            self.gamma,
            self.vol,
            self.dim[1],
            self.dim[2],
        ],
        device=self.device,
    )
```

### 2. 求解Euler方程
- **計算原始量**：在每個時間步計算保存的保守量和原始量。
- **計算通量**：計算流體在每個面上的通量，這些通量用於更新流體狀態。
- **應用通量**：將計算出的通量應用於保守量，更新流體狀態。

```python
def generate_batch(self) -> None:
    """Solve for new batch of simulations"""

    # initialize tensors with random coef
    self.initialize_batch()

    # run solver
    for s in range(self.nr_snapshots):
        # save arrays for
        wp.copy(self.seq_rho[s], self.rho)
        wp.copy(self.seq_vel[s], self.vel)
        wp.copy(self.seq_p[s], self.p)

        # iterations
        for i in range(self.iteration_per_snapshot):
            # compute primitives
            wp.launch(
                euler_conserved_to_primitive_batched_2d,
                dim=self.dim,
                inputs=[
                    self.mass,
                    self.mom,
                    self.e,
                    self.rho,
                    self.vel,
                    self.p,
                    self.gamma,
                    self.vol,
                    self.dim[1],
                    self.dim[2],
                ],
                device=self.device,
            )

            # compute extrapolations to faces
            wp.launch(
                euler_extrapolation_batched_2d,
                dim=self.dim,
                inputs=[
                    self.rho,
                    self.vel,
                    self.p,
                    self.rho_xl,
                    self.rho_xr,
                    self.rho_yl,
                    self.rho_yr,
                    self.vel_xl,
                    self.vel_xr,
                    self.vel_yl,
                    self.vel_yr,
                    self.p_xl,
                    self.p_xr,
                    self.p_yl,
                    self.p_yr,
                    self.gamma,
                    self.dx,
                    self.dt,
                    self.dim[1],
                    self.dim[2],
                ],
                device=self.device,
            )

            # compute fluxes
            wp.launch(
                euler_get_flux_batched_2d,
                dim=self.dim,
                inputs=[
                    self.rho_xl,
                    self.rho_xr,
                    self.rho_yl,
                    self.rho_yr,
                    self.vel_xl,
                    self.vel_xr,
                    self.vel_yl,
                    self.vel_yr,
                    self.p_xl,
                    self.p_xr,
                    self.p_yl,
                    self.p_yr,
                    self.mass_flux_x,
                    self.mass_flux_y,
                    self.mom_flux_x,
                    self.mom_flux_y,
                    self.e_flux_x,
                    self.e_flux_y,
                    self.gamma,
                    self.dim[1],
                    self.dim[2],
                ],
                device=self.device,
            )

            # apply fluxes
            wp.launch(
                euler_apply_flux_batched_2d,
                dim=self.dim,
                inputs=[
                    self.mass_flux_x,
                    self.mass_flux_y,
                    self.mom_flux_x,
                    self.mom_flux_y,
                    self.e_flux_x,
                    self.e_flux_y,
                    self.mass,
                    self.mom,
                    self.e,
                    self.dx,
                    self.dt,
                    self.dim[1],
                    self.dim[2],
                ],
                device=self.device,
            )
```

### 3. 生成和處理數據
- **數據轉換**：將生成的密度、速度和壓力場轉換為PyTorch張量，以便後續機器學習模型的訓練和測試。
- **數據歸一化**：如果指定了歸一化參數，則對數據進行歸一化處理，使其在特定範圍內。

```python
def __iter__(self) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Yields
    ------
    Iterator[Tuple[Tensor, Tensor]]
        Infinite iterator that returns a batch of timeseries with (density, velocity, pressure)
        fields of size [batch, seq_length, dim, resolution, resolution]
    """
    # infinite generator
    while True:
        # run simulation
        self.generate_batch()

        # return all samples generated before rerunning simulation
        batch_ind = [
            np.arange(self.nr_snapshots - self.seq_length)
            for _ in range(self.batch_size)
        ]
        for b_ind in batch_ind:
            np.random.shuffle(b_ind)
        for bb in range(self.nr_snapshots - self.seq_length):
            # run over batch to gather samples
            batched_seq_rho = []
            batched_seq_vel = []
            batched_seq_p = []
            for b in range(self.batch_size):
                # gather seq from each batch
                seq_rho = []
                seq_vel = []
                seq_p = []
                for s in range(self.seq_length):
                    # get variables
                    rho = wp.to_torch(self.seq_rho[batch_ind[b][bb] + s])[b]
                    vel = wp.to_torch(self.seq_vel[batch_ind[b][bb] + s])[b]
                    p = wp.to_torch(self.seq_p[batch_ind[b][bb] + s])[b]

                    # add channels
                    rho = torch.unsqueeze(rho, 0)
                    vel = torch.permute(vel, (2, 0, 1))
                    p = torch.unsqueeze(p, 0)

                    # normalize values
                    if self.normaliser is not None:
                        rho = (
                            rho - self.normaliser["density"][0]
                        ) / self.normaliser["density"][1]
                        vel = (
                            vel - self.normaliser["velocity"][0]
                        ) / self.normaliser["velocity"][1]
                        p = (p - self.normaliser["pressure"][0]) / self.normaliser[
                            "pressure"
                        ][1]

                    # store for producing seq
                    seq_rho.append(rho)
                    seq_vel.append(vel)
                    seq_p.append(p)

                # concat seq
                batched_seq_rho.append(torch.stack(seq_rho, axis=0))
                batched_seq_vel.append(torch.stack(seq_vel, axis=0))
                batched_seq_p.append(torch.stack(seq_p, axis=0))

            # CUDA graphs static copies
            if self.output_rho is None:
                # concat batches
                self.output_rho = torch.stack(batched_seq_rho, axis=0)
                self.output_vel = torch.stack(batched_seq_vel, axis=0)
                self.output_p

 = torch.stack(batched_seq_p, axis=0)
            else:
                self.output_rho.data.copy_(torch.stack(batched_seq_rho, axis=0))
                self.output_vel.data.copy_(torch.stack(batched_seq_vel, axis=0))
                self.output_p.data.copy_(torch.stack(batched_seq_p, axis=0))

            yield {
                "density": self.output_rho,
                "velocity": self.output_vel,
                "pressure": self.output_p,
            }

def __len__(self):
    return sys.maxsize
```

## KelvinHelmholtz2D模擬的意義

1. **研究和開發**：
   - 提供高質量的數據樣本，用於研究Kelvin-Helmholtz不穩定性現象，並開發新穎的數值方法和算法。

2. **機器學習模型訓練**：
   - 使用生成的數據訓練機器學習模型，這些模型可以用來預測不穩定性流動行為。

3. **實際應用**：
   - 在天體物理、氣象學、航空航天等實際工程領域中應用這些模型，提高預測準確性和決策效率。

## 總結

KelvinHelmholtz2D數據管道是一種高效的模擬工具，通過隨機生成初始條件並求解Euler方程，生成Kelvin-Helmholtz不穩定性問題的數據。這些數據可以用於訓練和測試機器學習模型，特別是針對不穩定性流動問題的研究和應用。
```
