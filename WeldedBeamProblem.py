import numpy as np 
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem  # 最適化問題を定義クラス

class WeldedBeamProblem(Problem):
    P = 6000  # 荷重（ポンド）
    L = 14    # 長さ（インチ）
    E = 30 * (10 ** 6)  # 弾性率（psi）
    G = 12 * (10 ** 6)  # せん断弾性係数（psi）
    tau_max = 13600     # 最大せん断応力（psi）
    sigma_max = 30000   # 最大正応力（psi）

    def __init__(self):
        super().__init__( # WeldedBeamProblemクラスからProblemクラスのメソッドの呼び出し
            n_var=4,       # 設計変数の数
            n_obj=2,       # 目的関数の数
            n_constr=5,    # 制約の数
            xl=np.array([0.125, 0.1, 0.1, 0.125]),   # 変数の下限
            xu=np.array([5.0, 10.0, 10.0, 5.0])     # 変数の上限
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = self.cost(x)  
        f2 = self.deflection(x)  

        # 制約計算部
        g = self.constraints(x)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g

    # ビームのコスト
    def cost(self, x):
        return 1.10471 * x[:, 0]**2 * x[:, 1] + 0.04811 * x[:, 2] * x[:, 3] * (14.0 + x[:, 1])

    # ビームのたわみ
    def deflection(self, x):
        return 4 * self.P * self.L**3 / (self.E * x[:, 3] * x[:, 2]**3)

    # ビームに関する制約
    def constraints(self, x):
        g1 = self.tau(x) - self.tau_max
        g2 = self.sigma(x) - self.sigma_max
        g3 = x[:, 0] - x[:, 3]
        g4 = 0.125 - x[:, 0]
        g5 = self.P - self.Pc(x)
        return np.column_stack([g1, g2, g3, g4, g5])

    # せん断応力
    def tau(self, x):
        return np.sqrt(
            (self.tau_d(x) ** 2)
            + (2 * self.tau_d(x) * self.tau_dd(x) * x[:, 1] / (2 * self.R(x)))
            + (self.tau_dd(x) ** 2)
        )

    # 直接せん断応力
    def tau_d(self, x):
        return self.P / (np.sqrt(2) * x[:, 0] * x[:, 1])

    # 曲げモーメントによるせん断応力
    def tau_dd(self, x):
        return self.P * (self.L + x[:, 1] / 2) * self.R(x) / self.J(x)

    # 慣性半径
    def R(self, x):
        return np.sqrt((x[:, 1] ** 2) / 4 + ((x[:, 0] + x[:, 2]) / 2) ** 2)

    # 極モーメント
    def J(self, x):
        return 2 * (np.sqrt(2) * x[:, 0] * x[:, 1] * ((x[:, 1] ** 2) / 12 + ((x[:, 0] + x[:, 2]) / 2) ** 2))

    # 正応力
    def sigma(self, x):
        return 6 * self.P * self.L / (x[:, 3] * (x[:, 2] ** 2))

    # 臨界荷重
    def Pc(self, x):
        return (4.013 * self.E * np.sqrt((x[:, 2] ** 2) * (x[:, 3] ** 6) / 36) / (self.L ** 2)) * (
            1 - x[:, 2] / (2 * self.L) * np.sqrt(self.E / (4 * self.G))
        )

def main():
    problem = WeldedBeamProblem()  # get_problem("welded_beam")
    algorithm = NSGA2(
        pop_size=20,  # population
        n_offsprings=10,  # 子孫
        eliminate_duplicates=True  # 重複個体を削除
    )
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 50),# 世代数制限による最適化終了条件
        seed=0,
        verbose=False # 詳細情報表示可否(=True)
    )

    plot = Scatter()
    plot.add(res.F, color="blue")
    plot.save('./WeldedBeamProblem.png')
    plot.show()


if __name__ == '__main__':
    main()