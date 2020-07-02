function [] = skeletonDisp(X3)

lines4disp = [
  0 1
  1 2
  2 3
  3 4
  1 5
  5 6
  6 7

  7 8

  8 9
  9 10
  10 11
  11 12

  8 13
  13 14
  14 15
  15 16

  8 17
  17 18
  18 19
  19 20

  8 21
  21 22
  22 23
  23 24

  8 25
  25 26
  26 27
  27 28

  4 29
  29 30
  30 31
  31 32
  32 33
  29 34
  34 35
  35 36
  36 37
  29 38
  38 39
  39 40
  40 41
  29 42
  42 43
  43 44
  44 45
  29 46
  46 47
  47 48
  48 49
];


X = X3;

[T, n] = size(X);
n3 = n / 3;


xs = X(:, 1:3:n);
ys = X(:, 2:3:n);
zs = X(:, 3:3:n);

xmin = min(min(xs));
xmax = max(max(xs));
ymin = min(min(ys));
ymax = max(max(ys));
zmin = min(min(zs));
zmax = max(max(zs));


d3 = 0;
  
for t = 1:T
    hold off
    
    %Ls = [];

    if d3    
      plot3(xmin, zmin, -ymin, 'x')
      hold on
      plot3(xmax, zmax, -ymax, 'x')
      hold on
    else
      plot(xmin, -ymin, 'x')
      hold on
      plot(xmax, -ymax, 'x')
      hold on
    end
    
    
    for i = 1:size(lines4disp, 1)
      a = lines4disp(i, 1);
      b = lines4disp(i, 2);
      
      %L = sqrt((X(t, 3 * a + 1) - X(t, 3 * b + 1)).^2 + (X(t, 3 * a + 2) - X(t, 3 * b + 2)).^2 + (X(t, 3 * a + 3) - X(t, 3 * b + 3)).^2);
      %Ls = [Ls L];
      
      if d3
        plot3([X(t, 3 * a + 1), X(t, 3 * b + 1)], [X(t, 3 * a + 3), X(t, 3 * b + 3)], -[X(t, 3 * a + 2), X(t, 3 * b + 2)], '*-')
      else
        plot([X(t, 3 * a + 1), X(t, 3 * b + 1)], -[X(t, 3 * a + 2), X(t, 3 * b + 2)], '.-')
      end
      hold on
      
    end
  
    hold off
    pause(0.001)
    
    %disp(Ls)
  
end
