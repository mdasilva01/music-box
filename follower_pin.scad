union() {
	cylinder($fn = 32, d = 4.0, h = 8);
	translate(v = [0, 0, -1]) {
		cube(center = true, size = [10, 10, 2]);
	}
}
