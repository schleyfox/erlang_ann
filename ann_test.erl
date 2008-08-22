-module(ann_test).
-export([setup/0, run/0]).

setup() ->
  ann_graph:start(),
  N1_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N2_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N3_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N4_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N5_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N6_pid = spawn(ann, perceptron, [[],[],[],[]]),
  N7_pid = spawn(ann, perceptron, [[],[],[],[]]),

  ann:connect(N1_pid, N2_pid),
  ann:connect(N1_pid, N3_pid),

  ann:connect(N2_pid, N4_pid),
  ann:connect(N2_pid, N5_pid),
  ann:connect(N2_pid, N6_pid),

  ann:connect(N3_pid, N4_pid),
  ann:connect(N3_pid, N5_pid),
  ann:connect(N3_pid, N6_pid),

  ann:connect(N4_pid, N7_pid),
  ann:connect(N5_pid, N7_pid),
  ann:connect(N6_pid, N7_pid),

  N1_pid ! {pass, 0.5, 0.7}.

run() -> 
  ann_graph:start(),
  X1_pid = spawn(ann, perceptron, [[],[],[]]),
  X2_pid = spawn(ann, perceptron, [[],[],[]]),
  H1_pid = spawn(ann, perceptron, [[],[],[]]),
  H2_pid = spawn(ann, perceptron, [[],[],[]]),

  O_pid = spawn(ann, perceptron,  [[],[],[]]),

  % Connect input node X1 to hidden nodes H1 and H2
  ann:connect(X1_pid, H1_pid),
  ann:connect(X1_pid, H2_pid),

  % Connect input node X2 to hidden nodes H1 and H2
  ann:connect(X2_pid, H1_pid),
  ann:connect(X2_pid, H2_pid),

  % Connect input node H1 and H2 to output node O
  ann:connect(H1_pid, O_pid),
  ann:connect(H2_pid, O_pid),

  X1_pid ! {status},
  X2_pid ! {status},
  H1_pid ! {status},
  H2_pid ! {status},
  O_pid ! {status},

  X1_pid ! {pass, 1.8},
  X2_pid ! {pass, 1.3}.
