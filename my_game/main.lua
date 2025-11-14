--[[
/**
 * main.lua: love.load documentation
 *
 * Purpose:
 *  This documents the love.load callback found in the project's main.lua.
 *  love.load is a special entry-point function used by the LÖVE (Love2D) game
 *  framework. It is invoked once when the game starts and is the appropriate
 *  place to perform one-time initialization.
 *
 * Typical responsibilities inside love.load:
 *  - Initialize global game state and variables (player state, level data, etc.).
 *  - Load resources (images, spritesheets, fonts, audio) using love.graphics
 *    and love.audio APIs (e.g. love.graphics.newImage, love.audio.newSource).
 *  - Configure window and graphics settings (e.g. love.window.setMode).
 *  - Initialize physics, input mappings, timers, and any external libraries.
 *  - Set up default values for gameplay constants and seed random number generators.
 *
 * Notes and best practices:
 *  - Keep heavy or blocking operations short; long-load tasks can block the
 *    main thread and delay startup. Consider streaming or showing a loading screen
 *    if many assets must be loaded.
 *  - Do not rely on love.update or love.draw being called before love.load finishes.
 *  - Use descriptive variable names for global state to avoid accidental shadowing.
 *  - Handle resource load failures gracefully (check for nil returns where applicable)
 *    and provide useful diagnostics for debugging.
 *
 * Example intent (not code):
 *  The love.load function in this file should prepare everything the game needs
 *  so that subsequent love.update and love.draw callbacks can run each frame.
 *
 * File location:
 *  /workspaces/BME3053C-Fall-2025/my_game/main.lua
 */
]]
function love.load()
  love.window.setTitle("Hello from Codespaces + LÖVE")
  width, height = 800, 600
  love.window.setMode(width, height)
  msg = "It works! 🚀  Use arrow keys to move the box."
  player = { x = 100, y = 100, s = 40, v = 200 }
end

function love.update(dt)
  if love.keyboard.isDown("d") then player.x = player.x + player.v * dt end
  if love.keyboard.isDown("a")  then player.x = player.x - player.v * dt end
  if love.keyboard.isDown("s")  then player.y = player.y + player.v * dt end
  if love.keyboard.isDown("w")    then player.y = player.y - player.v * dt end
end

function love.draw()
  love.graphics.print(msg, 20, 20)
  love.graphics.rectangle("fill", player.x, player.y, player.s, player.s)
end