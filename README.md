# quket
```
 Quantum Computing Simulator Ver Beta
     Copyright 2019-2020 Takashi Tsuchimochi, Yuto Mori, Takahiro Yoshikura. All rights Reserved.

 This suite of programs simulates quantum computing for electronic Hamiltonian.
 It currently supports the following methods:
   
   - Ground state VQE

```
# 必要なライブラリ

 - openfermion        0.10.0 
 - openfermionpyscf   0.4    
 - Qulacs             0.1.9   
 - numpy
 - scipy 

Ravel (ravel01) ではPython3.8が全ユーザー入っている。

Titan (titan2~titan7) では/home/calc/tsuchimochi/binをPATHに入れるように環境設定すれば使えるようになる。 


# 使い方

(1) インプットファイル `***.inp` を用意する (具体例は下か`main.py`を見る)

(2) 以下のコマンドで流す

```
     python3.8 main.py *** 
```
`***.log`の中に結果が出力される。

Titanで行う場合は、
```
     python3.8 main.py *** &
```
と`&`をつけるとログアウトしても実行し続ける。Ravelで行う場合は頭に`nohup`をつけると実行し続ける。


# ファイルの説明

- `***.inp` ：　入力ファイル
- `***.log` ：　出力ファイル
- `***.chk` ：　PySCFの積分やエネルギーなどの情報が入ってる
また、手法によっては以下のようなファイルが生成される
- `***.theta` ：　UCC法のt-amplitudes （VQEパラメータ）. 
- `***.kappa` ： 軌道変換のためのkappa-amplitudes （UCCのSinglesに対応）.
これらのファイルはinitial guessとして読み込むことが出来る。

# How to write `***.inp`

オプションを一行ずつ並べる。
いくつかのサンプルが`samples`ディレクトリにある。

## MINIMUM OPTIONS 
- `method`        : 手法を指定。VQEでは uhf, uccsd, sauccsd, phf, opt_puccd, など
- `geometry`      : 分子構造をXYZフォーマット（元素記号のあとにx,y,z座標）で指定。行の最初の文字が元素記号でない場合読み込みを終える
```
geometry
  A    Ax  Ay  Az
  B    Bx  By  Bz
  C    Cx  Cy  Cz 
  ...
```

- `n_electrons`   : 電子数 

- `n_orbitals`    : 空間


Inserting '@@@' in lines separates jobs. This enables multiple jobs with a single input file.
##### CAUTION!! The options from previous jobs remain the same unless redefined.

Options that have a default value (see main.py for details)

## For PySCF
- `basis`               :Gaussian Basis Set 
- `multiplicity`        :Spin multiplicity (defined as Nalpha - Nbeta + 1) 
- `charge`              :Electron charge (0 for neutral) 
- `pyscf_guess`         :Guess for pyscf: 'minao', 'chkfile'

## For qulacs (VQE part)
- `print_level`         :Printing level
- `mix_level`           :Number of pairs of orbitals to be mixed (to break symmetry)
- `rho`                 :Trotter number 
- `kappa_guess`         :Guess for kappa: 'zero', 'read', 'mix', 'random'
- `theta_guess`         :Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
- `Kappa_to_T1`         :Flag to use \*\*\*.kappa file (T1-like) for initial guess of T1
- `spin`                :Spin quantum number for spin-projection
- `ng`                  :Number of grid points for spin-projection
- `DS`                  :Ordering of T1 and T2: 0 for Exp[T1]Exp[T2], 1 for Exp[T2]Exp[T1]
- `print_amp_thres`     :Threshold for T amplitudes to be printed
- `constraint_lambda`   :Constraint for spin 

## For scipy.optimize
- `opt_method`          :Method for optimization
- `gtol`                :Convergence criterion based on gradient
- `ftol`                :Convergence criterion based on energy (cost)
- `eps`                 :Numerical step     
- `maxiter`             :Maximum iterations: if 0, skip VQE and only PySCF --> JW-transformation is carried out. 



# Requisites

The following external modules and libraries are required.
 - openfermion        0.10.0 
 - openfermionpyscf   0.4    
 - Qulacs             0.1.9   
 - numpy
 - scipy 

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

Options that have a default value (see main.py for details)

## For PySCF
- `basis`               :Gaussian Basis Set 
- `multiplicity`        :Spin multiplicity (defined as Nalpha - Nbeta + 1) 
- `charge`              :Electron charge (0 for neutral) 
- `pyscf_guess`         :Guess for pyscf: 'minao', 'chkfile'

## For qulacs (VQE part)
- `print_level`         :Printing level
