# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "SpecName"
  spec.version       = "4.1.0"
  spec.authors       = ["SpecAuthor"]
  spec.email         = ["specauthoremail9@gmail.com"]

  spec.summary       = "Spec Comments."
  spec.homepage      = "https://github.com"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r{^(assets|_layouts|_includes|LICENSE|README|feed|404|_data|tags|staticman)}i) }

  spec.metadata      = {
    "changelog_uri"     => "https://github.com/",
    "documentation_uri" => "https://github.com/"
  }

  spec.add_runtime_dependency "jekyll", "~> 3.8"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.4"
  spec.add_runtime_dependency "kramdown-parser-gfm", "~> 1.1"
  spec.add_runtime_dependency "kramdown", "~> 2.3.0"

  spec.add_development_dependency "bundler", ">= 1.16"
  spec.add_development_dependency "rake", "~> 12.0"
end
