function displayDigit (digit_unrolled)
  
  example_height = 20;
  example_width = 20;
  
  max_val = max(abs(digit_unrolled));
  digit = reshape(digit_unrolled, example_height, example_width) / max_val;
    
  colormap(gray);
  h = imagesc(digit);
  drawnow;
  
endfunction
