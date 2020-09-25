(defdfa left-right-no-dither ((1 2 3) (0 1 2 3 4) 0 (4))
          ((0 1 2)                              ;1 input token is the out of set token
           (1 2 3)
           (2 3 2)
           (3 4 3))
  ("Equivalent (roughly) to (lr){2}."))
