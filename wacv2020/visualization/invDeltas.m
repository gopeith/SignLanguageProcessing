function [X3] = invDeltas(x_deltas)

deltas = [
  0, 1  
  1, 2  
  2, 3  
  3, 4  
  1, 5  
  5, 6  
  6, 7  
  4, 29  
  29, 30  
  30, 31  
  31, 32  
  32, 33  
  29, 34  
  34, 35  
  35, 36  
  36, 37  
  29, 38  
  38, 39  
  39, 40  
  40, 41  
  29, 42  
  42, 43  
  43, 44  
  44, 45  
  29, 46  
  46, 47  
  47, 48  
  48, 49  
  7, 8  
  8, 9  
  9, 10  
  10, 11  
  11, 12  
  8, 13  
  13, 14  
  14, 15  
  15, 16  
  8, 17  
  17, 18  
  18, 19  
  19, 20  
  8, 21  
  21, 22  
  22, 23  
  23, 24  
  8, 25  
  25, 26  
  26, 27  
  27, 28  
];

ndeltas = size(deltas, 1);  


[T, dim] = size(x_deltas);

n = dim / 2;

X3 = [];
for t = 1:T

  X3(t, 1) = 0;
  X3(t, 2) = 0;
  X3(t, 3) = 0;
  
  for idelta = 1:ndeltas
    a = deltas(idelta, 1);
    b = deltas(idelta, 2);
  
    X3(t, 3 * b + 1) = X3(t, 3 * a + 1) - x_deltas(t, 2 * (idelta - 1) + 1);
    X3(t, 3 * b + 2) = X3(t, 3 * a + 2) - x_deltas(t, 2 * (idelta - 1) + 2);
  
  end

end
