tabout parity agele50 if muslim==1 using "Parity × Age≤50 | Muslim.csv"
foreach c in 30 40 50 {
  2.     di as text "===== Parity × Age≤`c' | MUSLIM ====="
  3.     tab parity agele`c' if muslim==1, col
  4.     di as text "===== Parity × Age≤`c' | NON-MUSLIM ====="
  5.     tab parity agele`c' if muslim==0, col
  6. }
