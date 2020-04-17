function Theta = randInitialise (m, n)
  epsilon = sqrt(6)/sqrt(m+n);
  Theta = 2*epsilon*rand(m, n)-epsilon;
endfunction
