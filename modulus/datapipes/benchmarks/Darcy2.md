# Darcy2D datapipes

Darcy2D data pipe 是一個數值模擬工具，用於生成二維Darcy流問題的數據。這些數據可以用來訓練和測試數據驅動模型，例如機器學習模型或神經算子（如PINN或FNO）。

## 核心功能

1. **生成隨機滲透率場**：
   - 使用隨機數生成傅里葉級數，然後將這些傅里葉級數轉換為滲透率場。這個過程確保了滲透率場的多樣性，使其可以代表不同的多孔介質結構。

2. **求解Darcy方程**：
   - 使用多重網格雅可比迭代法（或其他數值方法）求解Darcy方程。這個方程描述了多孔介質中流體的流動行為，通過求解這個方程可以得到流體在穩態條件下的壓力場分布。

3. **生成穩態壓力場**：
   - 通過數值求解得到最終的壓力場，這個壓力場反映了流體在多孔介質中的穩態流動狀況。

## 具體流程

### 1. 隨機生成滲透率場
- **隨機傅里葉級數生成**：使用隨機數生成傅里葉級數，這些級數用於構建初始的滲透率場。
- **傅里葉級數轉換**：將生成的傅里葉級數轉換為空間域的滲透率場。
- **閾值化處理**：對滲透率場進行閾值化處理，確保滲透率在指定範圍內，並且轉換為分段常數函數。

```python
def initialize_batch(self) -> None:
    """Initializes arrays for new batch of simulations"""
    self.permeability.zero_()
    seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
    wp.launch(
        kernel=init_uniform_random_4d,
        dim=self.fourier_dim,
        inputs=[self.rand_fourier, -1.0, 1.0, seed],
        device=self.device,
    )
    wp.launch(
        kernel=fourier_to_array_batched_2d,
        dim=self.dim,
        inputs=[
            self.permeability,
            self.rand_fourier,
            self.nr_permeability_freq,
            self.resolution,
            self.resolution,
        ],
        device=self.device,
    )
    wp.launch(
        kernel=threshold_3d,
        dim=self.dim,
        inputs=[
            self.permeability,
            0.0,
            self.min_permeability,
            self.max_permeability,
        ],
        device=self.device,
    )
    self.darcy0.zero_()
    self.darcy1.zero_()
