# Cierra Playground

Paste a program, drop a frame, see the grade and scopes live. The program's WGSL shader (the `cierra wgsl` target),
the output transform (display_encode exactly; aces2 as a 97^3 tetrahedral LUT) and the scopes run in the browser
on WebGPU; the local server compiles the program and decodes the frame.

    python tools/playground_server.py          # then open http://127.0.0.1:8790/

Needs a browser with WebGPU (Chrome / Edge 113+, Safari 26, Firefox 141+). Programs with spatial ops are refused with
the WGSL target's reason. Parity with the reference engine: tools/playground_parity.py.
