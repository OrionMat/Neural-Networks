function retval = displayDigitGrid(X)
  
  [m n] = size(X);
  example_height = 20;
  example_width = 20;
  
  display_rows = floor(sqrt(m));
  display_cols = ceil(m / display_rows);
  % between images padding
  pad = 1;
  
  % setup blank display
  display_array = - ones(pad + display_rows * (example_height + pad),...
                         pad + display_cols * (example_width + pad));
   
  % copy each example into a patch on the display array
  curr_ex = 1;
  for j = 1:display_rows
    for i = 1:display_cols
      if curr_ex > m, 
        break; 
      end
      max_val = max(abs(X(curr_ex, :)));
      display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                    pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                    reshape(X(curr_ex, :), example_height, example_width) / max_val;
      curr_ex = curr_ex + 1;
    end
    if curr_ex > m, 
      break; 
    end
  end

  colormap(gray);
  h = imagesc(display_array);
  axis image off
  drawnow;
  
endfunction