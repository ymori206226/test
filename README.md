# quket
```
 Quantum Computing Simulator Ver 0.4 (development version)
     Copyright 2019-2021 Takashi Tsuchimochi, Yuto Mori, Takahiro Yoshikura, Yoohee Ryo. All rights Reserved.

 This suite of programs simulates quantum computing for electronic Hamiltonian.
```
# 必要なライブラリ

 - openfermion        0.10.0 
 - openfermionpyscf   0.4    
 - Qulacs             0.1.9   
 - numpy
 - scipy 
 - mpi4py

Ravel (ravel01) ではPython3.8が全ユーザー入っている。

Titan (titan2~titan7) では/home/calc/tsuchimochi/binをPATHに入れるように環境設定すれば使えるようになる。 


# 使い方

(0) まず/work/USERNAME/dataのディレクトリを作る。

(1) インプットファイル `***.inp` を用意する (具体例は下か`main.py`を見る)

(2) 付属スクリプト`quket`を用いる。メインプログラム`main.py`があるディレクトリ$QDIRを`quket`内で指定し、

```
     ./quket -np $NPROCS *** 
```
NPROCSはMPI並列のプロセス数 (下記参照) で、`-np`オプションで指定する。毎回入力が面倒なら`quket`内で変数`nprocs`を書き換えれば省略できる。
ログアウトしても計算を続けるようにするには最後に`&`をつける。
`***.log`の中にシミュレーション結果が出力され、標準出力(stdout)は`***.out`に出力される。

スクリプトを用いない旧来の流し方は

```
     python3.8 main.py *** 
```

インプット内のnparによってOpenMPのスレッド並列計算が可能になるが、性能はCPUコア数に依存する。
無理なスレッド並列は逆に遅くなる要因となる。

VQEにおいてパラメータ数が多い場合、コスト関数の数値微分がネックとなる。
そこでMPI並列によってコスト関数の数値微分を並列計算する。MPI並列計算は`$NPROCS`をプロセス数として以下のように流す。

```
    mpirun -np $NPROCS python3.8 -m mpi4py main.py ***
```
これとOpenMPを組み合わせたOpenMP-MPIハイブリッド並列で大幅なスピードアップがのぞめる。


# ファイルの説明

- `***.inp` ：　入力ファイル
- `***.log` ：　出力ファイル
- `***.out` ：　標準出力ファイル (主に外部ライブラリが吐き出すコメント/警告/エラーなど)
- `***.chk` ：　PySCFの積分やエネルギーなどの情報が入ってる
また、手法によっては以下のようなファイルが生成される
- `***.theta` ：　UCC法のt-amplitudes （VQEパラメータ）. 
- `***.kappa` ： 軌道変換のためのkappa-amplitudes （UCCのSinglesに対応）.
これらのファイルはinitial guessとして読み込むことが出来る。

# How to write `***.inp`

オプションを一行ずつ並べる。
いくつかのサンプルが`samples`ディレクトリにある。

## 最低限必要なオプション
- `method`        : 手法を指定。VQEでは uhf, uccsd, sauccsd, phf, opt_puccd, など
- `geometry`      : 分子構造をXYZフォーマット（元素記号のあとにx,y,z座標）で指定。行の最初の文字が元素記号でない場合読み込みを終える。ただし、後述の通り`basis = hubbard`の場合は必要ない。
```
geometry
  A    Ax  Ay  Az
  B    Bx  By  Bz
  C    Cx  Cy  Cz 
  ...
```

- `n_electrons`   : 電子数 

- `n_orbitals`    : 用いる空間軌道の数（Qubitはこれの2倍）。HOMO-LUMO付近の軌道を取る


オプション間に '@@@' を挿入することでジョブを区切って実行することが出来る。
##### ただし、前のジョブで使ったオプションは新たに指定しなおさない限りそのまま引き継がれる


## その他のオプション

### For PySCF
- `basis`               :ガウス基底関数
- `multiplicity`        :スピン多重度 (Nalpha - Nbeta + 1) 
- `charge`              :電荷 (中性は0) 
- `pyscf_guess`         :pyscfにおけるguess: 'minao', 'chkfile'のいずれか

### For qulacs (VQE part)
- `print_level`         :出力レベル
- `mix_level`           :スピン非制限にするために混ぜるスピン軌道の数
- `rho`                 :トロッター数
- `kappa_guess`         :kappaのguess: 'zero', 'read', 'mix', 'random'のいずれか
- `theta_guess`         :T1 と T2 amplitudesのguess: 'zero', 'read', 'mix', 'random'のいずれか
- `Kappa_to_T1`         :`***.kappa`の内容を使ってT1を作るかの指定
- `DS`                  :T1とT2がかかる順番。 0 なら Exp[T1]Exp[T2], 1 なら Exp[T2]Exp[T1]
- `print_amp_thres`     :出力する T amplitudes のしきい値 
- `constraint_lambda`   :スピン制限のためのlambda値（大きければ正しいスピン多重度を与える）

### For scipy.optimize
- `opt_method`          :Minimizeにおける手法（L-BFGS等）
- `gtol`                :グラジエントの収束しきい値
- `ftol`                :エネルギーの収束しきい値
- `eps`                 :数値微分のステップサイズ     
- `maxiter`             :最大反復数: 0なら PySCFとJW-変換のみ行われて計算が終了, -1ならさらにエネルギー計算されて終了（どちらもVQEはしない）

### Parallel setting
- `npar`                :スレッド並列数の指定:デフォルトはスレッド並列なし（`npar`=1） 大体2か4くらいに設定すると良い. NumPyの都合上、最初に指定したものから変更できない

### Spin Symmetry Projection
- `spin`                :スピン射影におけるスピン多重度の指定
- `euler`               :スピン射影におけるオイラー角積分のグリッド点数. (alpha,beta,gamma)のデフォルトは(1,2,1). 通常はbetaだけの指定で十分なので、`euler`のインプットが1つならbetaの値, 2つならalphaとbetaの値, 3つなら全てが決まる.
```
euler =  4        ->  (1,4,1)
euler =  2, 4     ->  (2,4,1)
euler =  2, 4, 3  ->  (2,4,3)
```


### 初期行列式の決定
`det`もしくは`determinant`オプションによって初期ビット列を指定できる：デフォルトはHF配置
```
det = 00001111
```


### 多状態計算
`multi`セクションを使うことでJM-UCCが実行できる。左はゼロ次空間のビット列、右はエネルギーの重み
```
multi:
    00001111      0.5
    00110011      0.5
    (strings)   (weights)
``` 

### 励起状態計算
`excited`セクションを使うことでOrthogonally-Constrained VQE (OC-VQE) を使った励起状態計算が実行できる (UCCSD, k-UpCCGSDのみ対応)。
ビット列を指定すると、これを初期ビット列として複数の励起状態を段階的に探索する。下の例は2つの励起状態を`00001111`と`00100111`を出発点として探索する。このとき、初期ビット列は上から順に使われる。
```
excited:
    00001111
    00100111
```

# Requisites

The following external modules and libraries are required.
 - openfermion        0.10.0 
 - openfermionpyscf   0.4    
 - Qulacs             0.1.9   
 - numpy
 - scipy 
 - mpi4py

All the necessary libtaries are installed under Python3.8 in Ravel (ravel01) and Titan (titan2~titan7). 
Type "pip3.8 list" to check this.
You may run this program in your local machine too.


# How to use:

(1) Create an input file as `***.inp` (quick instruction below)

(2) Run main.py with python3.8

```
     python3.8 main.py *** 
```
The result is logged out in `***.log`.
If run on Ravel, it is recommended to add "nohup" option to prevent it from stopping when you log out the workstation (This is not necessary for Titan).



# File descriptions

- `***.inp` is an input file.
- `***.chk` contains integrals (and energy) from PySCF.
Depending on the method you choose in `***.inp`, there will be also `***.theta` and/or `***.kappa`. 
- `***.theta` stores t-amplitudes from UCC. 
- `***.kappa` stores kappa-amplitudes for orbital rotation.

You may read these files for initial guesses of subsequent calculations.




# How to write `***.inp`

Simply put options listed below.
The order does not matter.
Sample inputs are found in samples directory.


## MINIMUM OPTIONS 
- `method`        : method for VQE, either of  uhf, uccsd, sauccsd, phf, opt_puccd, etc.
- `geometry`      : a sequence of 'atom x y z' with a break.
- `n_electrons`   : number of electrons 
- `n_orbitals`    : number of spatial orbitals, Nqubit is twice this value


Inserting '@@@' in lines separates jobs. This enables multiple jobs with a single input file.
##### CAUTION!! The options from previous jobs remain the same unless redefined.

## Options that have a default value (see main.py for details)

### For PySCF
- `basis`               :Gaussian Basis Set 
- `multiplicity`        :Spin multiplicity (defined as Nalpha - Nbeta + 1) 
- `charge`              :Electron charge (0 for neutral) 
- `pyscf_guess`         :Guess for pyscf: 'minao', 'chkfile'

### For qulacs (VQE part)
- `print_level`         :Printing level
- `mix_level`           :Number of pairs of orbitals to be mixed (to break symmetry)
- `rho`                 :Trotter number 
- `kappa_guess`         :Guess for kappa: 'zero', 'read', 'mix', 'random'
- `theta_guess`         :Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
- `Kappa_to_T1`         :Flag to use `***.kappa` file (T1-like) for initial guess of T1
- `spin`                :Spin quantum number for spin-projection
- `ng`                  :Number of grid points for spin-projection
- `DS`                  :Ordering of T1 and T2: 0 for Exp[T1]Exp[T2], 1 for Exp[T2]Exp[T1]
- `print_amp_thres`     :Threshold for T amplitudes to be printed
- `constraint_lambda`   :Constraint for spin 

### For scipy.optimize
- `opt_method`          :Method for optimization
- `gtol`                :Convergence criterion based on gradient
- `ftol`                :Convergence criterion based on energy (cost)
- `eps`                 :Numerical step     
- `maxiter`             :Maximum iterations: if 0, skip VQE and only PySCF --> JW-transformation is carried out. 


