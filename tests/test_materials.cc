////////////////////////////////////////////////////////////////////////////////
#include <MeshFEM/Materials.hh>
#include <catch2/catch.hpp>
////////////////////////////////////////////////////////////////////////////////

using json = nlohmann::json;

////////////////////////////////////////////////////////////////////////////////

template<int N>
void test_material(const std::string &config) {
	// Test reading
	Materials::Constant<N> mat;
	mat.setFromJson(json::parse(config));
	std::cout << mat.getJson().dump(4) << std::endl;

	// Test writing and reading again
	json output1 = mat.getJson();
	Materials::Constant<N> mat2;
	mat2.setFromJson(output1);
	json output2 = mat2.getJson();
	REQUIRE(output1 == output2);
}

////////////////////////////////////////////////////////////////////////////////

TEST_CASE("reading material json files", "[materials]" ) {

	std::string config2d;
	std::string config3d;

	SECTION("Isotropic material") {
		config2d = config3d = R"({
			"type": "isotropic",
			"young": 200,
			"poisson": 0.3
		})";
	}

	SECTION("Othrotropic material") {
		// 2D: "young": [young_x, young_y],
		//     "poisson": [poisson_xy, poisson_yx],
		//     "shear": [shear_xy],
		config2d = R"({
			"type": "orthotropic",

			"young":   [2.933545, 2.933545],
			"poisson": [0.27186, 0.27186],
			"shear":   [0.87212]
		})";

		// 3D: "young": [young_x, young_y, young_z],
		//     "poisson": [poisson_yz, poisson_zy,
		//                 poisson_zx, poisson_xz,
		//                 poisson_xy, poisson_yx],
		//     "shear": [shear_yz, shear_zx, shear_xy],
		config3d = R"({
			"type": "orthotropic",
			"young": [1.0, 2.0, 3.0],
			"poisson": [0.6, 0.9, 0.9, 0.3, 0.3, 0.6],
			"shear": [0.1, 0.2, 0.3]
		})";
	}

	SECTION("Anisotropic material") {
		// "material_matrix": [[C_00, C_01, C02, C03, C04, C05],
		//                     [C_10, C_11, C12, C13, C14, C15],
		//                     [C_20, C_21, C22, C23, C24, C25],
		//                     [C_30, C_31, C32, C33, C34, C35],
		//                     [C_40, C_41, C42, C43, C44, C45],
		//                     [C_50, C_51, C52, C53, C54, C55]],
		config2d = R"({
			"type": "anisotropic",
			"material_matrix": [
				[9.0, 0.1, 0.2],
				[0.1, 9.0, 0.3],
				[0.2, 0.3, 1.0]
			]
		})";

		config3d = R"({
			"type": "anisotropic",
			"material_matrix": [
				[9.0, 0.1, 0.2, 0.5, 0.5, 0.5],
				[0.1, 9.0, 0.3, 0.5, 0.5, 0.5],
				[0.2, 0.3, 9.0, 0.5, 0.5, 0.5],
				[0.5, 0.5, 0.5, 1.5, 0.1, 0.2],
				[0.5, 0.5, 0.5, 0.1, 1.6, 0.3],
				[0.5, 0.5, 0.5, 0.2, 0.3, 1.7]
			]
		})";
	}

	test_material<2>(config2d);
	test_material<3>(config3d);
}

TEST_CASE("setting material bounds", "[materials]" ) {
	const std::string config = R"({
		"lower": [ [0,  25], [1, 0.1] ],
		"upper": [ [0, 292], [1, 0.6] ]
	})";

	Materials::Isotropic<2>::setBoundsFromJson(json::parse(config));
	Materials::Orthotropic<2>::setBoundsFromJson(json::parse(config));
	Materials::Isotropic<3>::setBoundsFromJson(json::parse(config));
	Materials::Orthotropic<3>::setBoundsFromJson(json::parse(config));
}
